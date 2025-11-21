"""
Download top-N arXiv-hosted PDFs by citation count using OpenAlex + arXiv PDFs.

Main functions:
- fetch_top_arxiv_works_from_openalex(...) -> list[dict]
- download_pdfs_from_metadata(...) -> None

Requirements:
- requests
- tqdm
- typing (stdlib)
- python 3.9+

Notes:
- Resumable: downloaded filenames/CSV metadata are saved; existing PDF files are skipped.
- Use `show_progress=False` to disable tqdm progress bars.
- Respect arXiv rate limits using `arxiv_delay_seconds` (default 3s). Adjust if you have explicit permission.
"""

from __future__ import annotations
import os
import re
import time
import csv
import json
import logging
from typing import Any, Dict, List, Optional, Iterable
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# ---------------------------
# Configuration / constants
# ---------------------------
OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_WORKS_ENDPOINT = f"{OPENALEX_BASE}/works"
# maximum per_page that OpenAlex supports reasonably; keep conservative
OPENALEX_PER_PAGE = 200
# HTTP session defaults
HTTP_TIMEOUT = 30  # seconds
# CSV metadata file
METADATA_CSV = "openalex_arxiv_top_metadata.csv"
# Regular expressions to extract arXiv ids from URLs
ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.IGNORECASE)
# chunk size for streaming downloads
STREAM_CHUNK_SIZE = 1 << 14  # 16 KB

# ---------------------------
# Utilities
# ---------------------------
def make_session() -> requests.Session:
    """Create a requests Session with retries."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """Try to extract an arXiv identifier from a URL or string.

    Returns normalized id like '1234.56789v1' or 'hep-th/9901001v1' or None.
    """
    m = ARXIV_ID_RE.search(url)
    if not m:
        return None
    return m.group(1)

def safe_filename(s: str) -> str:
    """Make a filesystem-safe filename."""
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)[:240]

# ---------------------------
# OpenAlex querying
# ---------------------------
def fetch_top_arxiv_works_from_openalex(
    top_n: int = 10_000,
    per_page: int = OPENALEX_PER_PAGE,
    mailto: Optional[str] = None,
    show_progress: bool = True,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch up to `top_n` OpenAlex works that are hosted on arXiv, ordered by cited_by_count desc.

    Strategy:
    - Use OpenAlex works endpoint with a filter that targets repository locations whose source name includes 'arXiv'.
    - Sort by cited_by_count:desc.
    - For each returned work, locate an arXiv URL or arXiv id inside `locations` or `best_oa_location`.
    - Return a list of metadata dicts with keys: openalex_id, cited_by_count, title, year, arxiv_id, pdf_url.
    """
    if session is None:
        session = make_session()

    collected: List[Dict[str, Any]] = []
    page = 1
    params_base = {
        "per-page": per_page,
        "sort": "cited_by_count:desc",
    }
    if mailto:
        params_base["mailto"] = mailto

    # Try a targeted filter: repository locations whose source display name has 'arXiv'
    # If OpenAlex accepts the filter "locations.source.display_name.search:arXiv" this will narrow results.
    # This is robust to multiple variants because we will scan each work's locations for arXiv URLs anyway.
    filter_expr = "locations.source.display_name.search:arXiv,locations.source.type:repository"
    params_base["filter"] = filter_expr

    pbar = None
    if show_progress:
        pbar = tqdm(total=top_n, desc="fetching OpenAlex pages", unit="works")

    while len(collected) < top_n:
        params = {**params_base, "page": page}
        try:
            resp = session.get(OPENALEX_WORKS_ENDPOINT, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logging.exception("OpenAlex request failed on page %d: %s", page, e)
            # back off and retry a few seconds
            time.sleep(5)
            continue

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            # locate an arXiv URL/id inside work['locations'] or best_oa_location or primary_location
            found_arxiv_id: Optional[str] = None
            candidate_pdf: Optional[str] = None

            # check best_oa_location and primary_location first (they may be present)
            for loc_field in ("best_oa_location", "primary_location"):
                loc = work.get(loc_field)
                if loc:
                    url = loc.get("url")
                    if url:
                        aid = extract_arxiv_id_from_url(url)
                        if aid:
                            found_arxiv_id = aid
                            if "pdf" in url or url.endswith(".pdf"):
                                candidate_pdf = url if url.endswith(".pdf") else f"https://arxiv.org/pdf/{aid}.pdf"
                            else:
                                candidate_pdf = f"https://arxiv.org/pdf/{aid}.pdf"

            # check all locations if not yet found
            if not found_arxiv_id and work.get("locations"):
                for loc in work["locations"]:
                    url = loc.get("url") or ""
                    aid = extract_arxiv_id_from_url(url)
                    if aid:
                        found_arxiv_id = aid
                        if url.endswith(".pdf"):
                            candidate_pdf = url
                        else:
                            candidate_pdf = f"https://arxiv.org/pdf/{aid}.pdf"
                        break

            # As a last resort, check identifiers in the `ids` map (some OpenAlex records may include an 'arXiv' external id)
            if not found_arxiv_id:
                ids_map = work.get("ids", {})
                if ids_map:
                    # Some OpenAlex records may include keys like 'arXiv' or 'arxiv'
                    # Try to find any value that looks like an arXiv id
                    for v in ids_map.values():
                        if isinstance(v, str):
                            aid = extract_arxiv_id_from_url(v)
                            if aid:
                                found_arxiv_id = aid
                                candidate_pdf = f"https://arxiv.org/pdf/{aid}.pdf"
                                break
                        elif isinstance(v, list):
                            for vv in v:
                                if isinstance(vv, str):
                                    aid = extract_arxiv_id_from_url(vv)
                                    if aid:
                                        found_arxiv_id = aid
                                        candidate_pdf = f"https://arxiv.org/pdf/{aid}.pdf"
                                        break
                            if found_arxiv_id:
                                break

            if found_arxiv_id:
                meta = {
                    "openalex_id": work.get("id"),
                    "title": work.get("display_name"),
                    "year": work.get("publication_year"),
                    "cited_by_count": work.get("cited_by_count", 0),
                    "arxiv_id": found_arxiv_id,
                    "pdf_url": candidate_pdf or f"https://arxiv.org/pdf/{found_arxiv_id}.pdf",
                }
                collected.append(meta)
                if pbar:
                    pbar.update(1)
                if len(collected) >= top_n:
                    break

        # pagination / cursor check
        meta = data.get("meta", {})
        total_found = meta.get("count", None)
        # If there are no more pages, break
        if not data.get("meta") or not data.get("results"):
            break

        page += 1
        # Be polite to OpenAlex - they allow higher rates, but keep modest pacing
        time.sleep(0.12)  # small pause between OpenAlex pages

    if pbar:
        pbar.close()

    # ensure deduplication by arXiv id, keep the highest-cited occurrence
    dedup: Dict[str, Dict[str, Any]] = {}
    for m in collected:
        aid = m["arxiv_id"]
        existing = dedup.get(aid)
        if not existing or (m["cited_by_count"] > existing["cited_by_count"]):
            dedup[aid] = m

    results_final = sorted(dedup.values(), key=lambda x: x["cited_by_count"], reverse=True)[:top_n]
    return results_final

# ---------------------------
# PDF downloading
# ---------------------------
def download_pdfs_from_metadata(
    metas: Iterable[Dict[str, Any]],
    out_dir: str = "arxiv_pdfs",
    arxiv_delay_seconds: float = 3.0,
    show_progress: bool = True,
    session: Optional[requests.Session] = None,
    metadata_csv: str = METADATA_CSV,
) -> None:
    """
    Download PDFs given metadata entries with 'arxiv_id' and 'pdf_url'. Resumable.

    - Files are saved as "<arxiv_id>.pdf" (sanitized).
    - metadata_csv is updated/appended so you can track progress & resume.
    - Downloads are sequential and wait `arxiv_delay_seconds` between requests (default 3s).
    """
    os.makedirs(out_dir, exist_ok=True)
    if session is None:
        session = make_session()

    # Load already-saved metadata to avoid duplicates and maintain mapping
    existing_meta: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(metadata_csv):
        try:
            with open(metadata_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_meta[row["arxiv_id"]] = row
        except Exception:
            # ignore and overwrite later
            existing_meta = {}

    metas_list = list(metas)
    pbar = tqdm(metas_list, desc="downloading PDFs", disable=not show_progress, unit="file")

    for row in pbar:
        arxiv_id = row["arxiv_id"]
        fname = safe_filename(f"{arxiv_id}.pdf")
        target_path = os.path.join(out_dir, fname)

        # skip if file exists and non-zero
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
            pbar.set_postfix_str(f"skipping {arxiv_id} (exists)")
            # ensure the metadata is recorded
            if arxiv_id not in existing_meta:
                existing_meta[arxiv_id] = {
                    "openalex_id": row.get("openalex_id", ""),
                    "title": row.get("title", ""),
                    "year": str(row.get("year", "")),
                    "cited_by_count": str(row.get("cited_by_count", "")),
                    "arxiv_id": arxiv_id,
                    "pdf_url": row.get("pdf_url", ""),
                    "file_path": target_path,
                }
            time.sleep(arxiv_delay_seconds)
            continue

        pdf_url = row.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        # ensure URL is normalized to https arxiv.org/pdf/{id}.pdf when possible
        if "arxiv.org" in pdf_url and not pdf_url.endswith(".pdf"):
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        success = False
        try:
            with session.get(pdf_url, stream=True, timeout=HTTP_TIMEOUT) as resp:
                if resp.status_code == 200 and resp.headers.get("Content-Type", "").lower().startswith("application/pdf"):
                    total = int(resp.headers.get("Content-Length") or 0)
                    # write stream
                    with open(target_path, "wb") as out_f:
                        if show_progress and total > 0:
                            inner = tqdm(total=total, unit="B", unit_scale=True, leave=False)
                        else:
                            inner = None
                        for chunk in resp.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                            if chunk:
                                out_f.write(chunk)
                                if inner:
                                    inner.update(len(chunk))
                        if inner:
                            inner.close()
                    success = True
                else:
                    # try to detect if server returned HTML that redirects or blocked
                    text = resp.text[:400] if resp.text else ""
                    logging.warning("Non-PDF response for %s: status=%s content-type=%s snippet=%s", pdf_url, resp.status_code, resp.headers.get("Content-Type"), text)
        except Exception as e:
            logging.exception("Failed to download %s -> %s", pdf_url, e)
            success = False

        # If failed, try a fallback arXiv PDF URL classic pattern
        if not success and "arxiv.org" not in pdf_url:
            fallback = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            try:
                with session.get(fallback, stream=True, timeout=HTTP_TIMEOUT) as resp:
                    if resp.status_code == 200 and resp.headers.get("Content-Type", "").lower().startswith("application/pdf"):
                        with open(target_path, "wb") as out_f:
                            for chunk in resp.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                                if chunk:
                                    out_f.write(chunk)
                        success = True
                    else:
                        logging.warning("Fallback also failed for %s (status %s)", fallback, resp.status_code)
            except Exception as e:
                logging.exception("Fallback download error for %s: %s", fallback, e)
                success = False

        # record metadata even if failed (user can inspect)
        existing_meta[arxiv_id] = {
            "openalex_id": row.get("openalex_id", ""),
            "title": row.get("title", ""),
            "year": str(row.get("year", "")),
            "cited_by_count": str(row.get("cited_by_count", "")),
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "file_path": target_path if success else "",
            "downloaded": "1" if success else "0",
        }

        # polite delay after each download attempt (arXiv guidance: no more than 1 req every 3s)
        time.sleep(arxiv_delay_seconds)

    pbar.close()

    # write out metadata CSV (overwrite with current state)
    fieldnames = ["openalex_id", "title", "year", "cited_by_count", "arxiv_id", "pdf_url", "file_path", "downloaded"]
    try:
        with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for aid, md in existing_meta.items():
                # ensure all fields exist
                row = {k: md.get(k, "") for k in fieldnames}
                writer.writerow(row)
    except Exception:
        logging.exception("Failed to write metadata CSV to %s", metadata_csv)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # PARAMETERS YOU CAN CHANGE
    TOP_N = 100  # change for testing (e.g., 50) then set to 10_000 for full run
    OUT_DIR = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs"
    OPENALEX_MAILTO = None  # optionally supply an email to be a good API citizen, e.g., "you@example.com"
    SHOW_PROGRESS = True
    ARXIV_DELAY_SECONDS = 3.0  # recommended >= 3.0 to respect arXiv API Terms of Use

    sess = make_session()

    print("Step 1: fetching metadata from OpenAlex (this may take many pages)...")
    metas = fetch_top_arxiv_works_from_openalex(
        top_n=TOP_N,
        per_page=OPENALEX_PER_PAGE,
        mailto=OPENALEX_MAILTO,
        show_progress=SHOW_PROGRESS,
        session=sess,
    )

    print(f"Retrieved {len(metas)} arXiv-hosted works from OpenAlex (top by citation). Sample:")
    for m in metas[:5]:
        print(json.dumps({"arxiv_id": m["arxiv_id"], "cited_by_count": m["cited_by_count"], "title": m["title"]}, ensure_ascii=False))

    print("Step 2: downloading PDFs (sequential; resumable).")
    download_pdfs_from_metadata(
        metas,
        out_dir=OUT_DIR,
        arxiv_delay_seconds=ARXIV_DELAY_SECONDS,
        show_progress=SHOW_PROGRESS,
        session=sess,
        metadata_csv=METADATA_CSV,
    )

    print("Done. Check the directory and metadata CSV for status and resume support.")
