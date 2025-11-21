"""
Top-10k arXiv PDF downloader (OpenAlex-backed)

How it works:
1. Find the OpenAlex source entry whose display name mentions "arXiv".
2. Query OpenAlex /works with filter=repository:<SOURCE_ID>, sorted by cited_by_count:desc.
3. Page the API (per-page up to 200) until we have desired N works.
4. For each work, try to find a PDF URL (best_oa_location.pdf_url, locations[*].pdf_url).
   If missing, attempt fallback to the arXiv PDF pattern: https://arxiv.org/pdf/<arXiv_id>.pdf
5. Download PDFs to the output directory in parallel.
"""

from __future__ import annotations
import os
import time
import math
import json
import logging
from typing import Any, Dict, Generator, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests import Response
from tqdm import tqdm

# -------------------------
# CONFIGURATION (edit here)
# -------------------------
OPENALEX_BASE = "https://api.openalex.org"
# Optional: include your email in OpenAlex calls with &mailto=your_email@example.com to be polite
POLITE_EMAIL = None  # e.g. "your_email@example.com" or leave None

# Number of top PDFs you want
TARGET_N = 200

# Where to save PDFs
OUT_DIR = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs"

# How many parallel downloads (adjust to your bandwidth and politeness)
MAX_DOWNLOAD_WORKERS = 12

# Per-page when querying OpenAlex: between 1 and 200 (docs say up to 200)
PER_PAGE = 200

# Toggle progress bars
show_progress: bool = True  # set to False to disable tqdm bars

# Timeout and retry settings
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------------------------
# Helpers
# -------------------------
def polite_url(url: str) -> str:
    """Append mailto if configured so OpenAlex can put you in the polite pool."""
    if POLITE_EMAIL:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}mailto={POLITE_EMAIL}"
    return url


def req_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GET JSON with simple retry/backoff on common transient errors (429, 5xx)."""
    tries = 0
    while True:
        tries += 1
        try:
            full_url = polite_url(url)
            r: Response = requests.get(full_url, params=params or {}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504) and tries <= MAX_RETRIES:
                sleep_for = BACKOFF_FACTOR ** tries
                logging.warning("Transient HTTP %s for %s — sleeping %.1fs then retrying (try %d)",
                                r.status_code, url, sleep_for, tries)
                time.sleep(sleep_for)
                continue
            r.raise_for_status()
        except requests.RequestException as ex:
            if tries <= MAX_RETRIES:
                sleep_for = BACKOFF_FACTOR ** tries
                logging.warning("RequestException %s — sleeping %.1fs then retrying (try %d)", ex, sleep_for, tries)
                time.sleep(sleep_for)
                continue
            logging.error("Failed GET %s after %d tries: %s", url, tries, ex)
            raise


def req_get_stream(url: str) -> Response:
    """GET a streaming response (for downloading PDFs), with a few retries."""
    tries = 0
    while True:
        tries += 1
        try:
            full_url = polite_url(url)
            r = requests.get(full_url, stream=True, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 502, 503, 504) and tries <= MAX_RETRIES:
                time.sleep(BACKOFF_FACTOR ** tries)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if tries <= MAX_RETRIES:
                time.sleep(BACKOFF_FACTOR ** tries)
                continue
            raise


# -------------------------
# OpenAlex helpers
# -------------------------
def find_arxiv_source_id() -> Optional[str]:
    """
    Find the OpenAlex source ID for arXiv by searching sources.
    Returns the source ID (like 'https://openalex.org/Sxxxxx') or None.
    """
    url = f"{OPENALEX_BASE}/sources"
    params = {"filter": "display_name.search:arXiv", "per-page": 200}
    data = req_get_json(url, params=params)
    results = data.get("results", [])
    # pick source whose display_name contains 'arXiv' (case-insensitive)
    for src in results:
        name = src.get("display_name", "")
        if "arxiv" in name.lower():
            return src.get("id")
    # fallback: if none matched exactly, return first if any
    if results:
        return results[0].get("id")
    return None


def stream_arxiv_hosted_works(repository_source_id: str, target_n: int = TARGET_N) -> Generator[Dict[str, Any], None, None]:
    """
    Stream works from OpenAlex that are hosted in the given repository (OpenAlex source ID),
    sorted by cited_by_count descending (i.e., greatest citations first).
    Yields OpenAlex work objects until target_n is reached or API exhausted.
    """
    page = 1
    collected = 0
    # filter param: repository:<source_id> (docs allow repository filter)
    base_url = f"{OPENALEX_BASE}/works"
    filter_param = f"repository:{repository_source_id}"
    # sort by cited_by_count descending
    sort_param = "cited_by_count:desc"
    pbar = None
    if show_progress:
        pbar = tqdm(total=target_n, desc="collecting works", unit="works")
    while collected < target_n:
        params = {
            "filter": filter_param,
            "sort": sort_param,
            "per-page": PER_PAGE,
            "page": page,
        }
        data = req_get_json(base_url, params=params)
        results = data.get("results", [])
        if not results:
            break
        for work in results:
            yield work
            collected += 1
            if pbar:
                pbar.update(1)
            if collected >= target_n:
                break
        page += 1
        # safety: stop if there are no more pages
        meta = data.get("meta", {})
        if meta.get("page") is None or meta.get("page") >= meta.get("total_pages", math.inf):
            break
    if pbar:
        pbar.close()


def extract_best_pdf_url_from_work(work: Dict[str, Any]) -> Optional[str]:
    """
    Attempt to extract the most direct PDF URL from an OpenAlex work object.
    Strategy:
      1. best_oa_location.pdf_url
      2. locations[*].pdf_url (first)
      3. landing_page_url that ends with .pdf
      4. fallback to arXiv pattern if an arXiv id is present in work['ids']
    """
    # 1
    bo = work.get("best_oa_location")
    if bo:
        pdf = bo.get("pdf_url") or bo.get("url_for_landing_page")
        if pdf:
            return pdf
    # 2
    locs = work.get("locations", []) or []
    for loc in locs:
        if loc.get("pdf_url"):
            return loc.get("pdf_url")
        if loc.get("url_for_landing_page") and loc.get("url_for_landing_page").lower().endswith(".pdf"):
            return loc.get("url_for_landing_page")
    # 3
    if work.get("id"):
        # sometimes the landing page in 'id' is a DOI URL; skip.
        pass
    landing = work.get("id")  # not generally a pdf
    # 4. check external ids for arXiv
    ids = work.get("ids", {})
    # ids may include arXiv entries; also check 'alternate_hosts' or 'x' fields
    # OpenAlex often stores external IDs under 'ids' like {"arxiv": "2301.01234"}
    if isinstance(ids, dict):
        arxiv_id = ids.get("arxiv") or ids.get("arXiv") or ids.get("arXivId")
        if arxiv_id:
            # sanitize
            aid = arxiv_id.split(":")[-1]
            return f"https://arxiv.org/pdf/{aid}.pdf"
    # fallback: sometimes the OpenAlex 'id' field contains a list of identifiers in other places - also check work['display_name']
    # As a last try, try to find an arXiv id inside the 'alternate_host_venues' or 'sources' (not guaranteed).
    # Give up if none found
    return None


# -------------------------
# Downloading
# -------------------------
def download_pdf_task(item: Tuple[int, str, str]) -> Tuple[int, str, bool, Optional[str]]:
    """
    Download task for ThreadPoolExecutor.
    Returns (index, filename, success, error_message)
    item: (index, pdf_url, out_path)
    """
    idx, pdf_url, out_path = item
    try:
        r = req_get_stream(pdf_url)
        # some hosts send wrong content types; still write the bytes
        total = r.headers.get("content-length")
        if total is None:
            # unknown size: write directly
            with open(out_path, "wb") as fh:
                for chunk in r.iter_content(8192):
                    if chunk:
                        fh.write(chunk)
        else:
            total_i = int(total)
            if show_progress:
                pbar = tqdm(total=total_i, unit="B", unit_scale=True, desc=f"dl {os.path.basename(out_path)}")
            else:
                pbar = None
            with open(out_path, "wb") as fh:
                for chunk in r.iter_content(8192):
                    if chunk:
                        fh.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))
            if pbar:
                pbar.close()
        return (idx, out_path, True, None)
    except Exception as ex:
        return (idx, out_path, False, str(ex))


def download_pdfs(pdf_items: List[Tuple[int, str, str]]) -> List[Tuple[int, str, bool, Optional[str]]]:
    """
    pdf_items: list of tuples (index, pdf_url, outfile_path)
    Downloads them in parallel (ThreadPool) and returns results list.
    """
    results: List[Tuple[int, str, bool, Optional[str]]] = []
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as exe:
        futures = {exe.submit(download_pdf_task, it): it for it in pdf_items}
        if show_progress:
            main_pbar = tqdm(total=len(futures), desc="downloading PDFs")
        else:
            main_pbar = None
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if main_pbar:
                main_pbar.update(1)
        if main_pbar:
            main_pbar.close()
    return results


# -------------------------
# Orchestration
# -------------------------
def run_pipeline(target_n: int = TARGET_N, out_dir: str = OUT_DIR) -> None:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # 1. Find arXiv source id on OpenAlex
    logging.info("Finding OpenAlex source id for arXiv...")
    source_id = find_arxiv_source_id()
    if not source_id:
        logging.error("Could not find arXiv source in OpenAlex sources. Aborting.")
        return
    logging.info("Found OpenAlex source id: %s", source_id)

    # 2. Stream top works hosted on that repository, sorted by cited_by_count desc
    logging.info("Collecting top %d works hosted on %s ...", target_n, source_id)
    works_gen = stream_arxiv_hosted_works(source_id, target_n)

    # 3. For each work extract pdf url (and metadata), collect until we have target_n pdf candidates
    pdf_items: List[Tuple[int, str, str]] = []
    skipped = 0
    idx = 0
    for work in works_gen:
        if idx >= target_n:
            break
        idx += 1
        pdf_url = extract_best_pdf_url_from_work(work)
        # If no explicit PDF url, try to find arXiv id embedded in ``ids`` or other fields:
        if not pdf_url:
            # attempt parse from 'external_ids' or 'openalex' fields
            ids = work.get("ids", {})
            # attempt common patterns
            arxiv_id = None
            if isinstance(ids, dict):
                arxiv_id = ids.get("arxiv") or ids.get("arXiv")
            # sometimes 'display_name' or 'title' won't help. skip if none.
            if arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if not pdf_url:
            skipped += 1
            continue
        # create safe filename
        title_snippet = work.get("display_name", "")[:80].strip().replace("/", "_").replace("\\", "_")
        cited_by = work.get("cited_by_count", 0)
        # prefer using arXiv id if available
        safe_id = None
        ids = work.get("ids", {})
        if isinstance(ids, dict):
            safe_id = ids.get("arxiv")
        if not safe_id:
            # fallback: make a short safe id from work id
            safe_id = work.get("id", "").split("/")[-1] or f"work{idx}"
        filename = f"{idx:06d}__{safe_id}__c{cited_by}__{title_snippet}.pdf"
        outfile = os.path.join(out_dir, filename)
        pdf_items.append((idx, pdf_url, outfile))

    logging.info("Collected %d PDF candidates (skipped %d works without PDFs).", len(pdf_items), skipped)

    # 4. Download files in parallel
    logging.info("Starting downloads (%d files) ...", len(pdf_items))
    results = download_pdfs(pdf_items)
    succ = sum(1 for _i, _path, ok, _err in results if ok)
    fail = len(results) - succ
    logging.info("Downloads completed: %d succeeded, %d failed", succ, fail)
    if fail:
        logging.info("Failed downloads (first 10 reported):")
        c = 0
        for idx, path, ok, err in results:
            if not ok:
                logging.info("  %d: %s -> %s", idx, path, err)
                c += 1
                if c >= 10:
                    break


# -------------------------
# Run as script
# -------------------------
if __name__ == "__main__":
    # small example: if you want to test with fewer items:
    # run_pipeline(target_n=100, out_dir="test_pdfs")
    run_pipeline()
