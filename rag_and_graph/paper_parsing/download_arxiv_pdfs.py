import requests
import time
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm

def get_top_arxiv_papers_metadata(
    n_papers: int = 200,
    email: str = "your_email@example.com",
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetches metadata for the top N most cited papers that have an arXiv version.
    Handles pagination automatically to retrieve more than 200 results.
    """
    base_url = "https://api.openalex.org/works"
    arxiv_source_id = "S4306400194"

    # We use cursor pagination to efficiently get >200 results
    params = {
        "filter": f"locations.source.id:{arxiv_source_id}",
        "sort": "cited_by_count:desc",
        "per-page": 200,
        "mailto": email,
        "cursor": "*"  # Start cursor
    }

    all_papers: List[Dict[str, Any]] = []

    # Progress bar for metadata fetching (estimated pages)
    total_pages = (n_papers + 199) // 200
    pbar = tqdm(total=n_papers, desc="Fetching Metadata", disable=not show_progress, unit="paper")

    while len(all_papers) < n_papers:
        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            if not results:
                break  # No more results available

            for work in results:
                if len(all_papers) >= n_papers:
                    break

                # Extract arXiv specific PDF URL
                arxiv_pdf_url: Optional[str] = None
                for location in work.get('locations', []):
                    # Check if this location is arXiv
                    src = location.get('source')
                    if src and isinstance(src, dict) and src.get('id') and str(src.get('id')).endswith(arxiv_source_id):
                        arxiv_pdf_url = location.get('pdf_url')
                        break

                # Fallback: if no arXiv PDF URL, skip this result (we only want arXiv pdfs)
                if not arxiv_pdf_url:
                    continue

                # Attempt to get the arXiv id from OpenAlex IDs
                arxiv_id: Optional[str] = None
                ids = work.get('ids', {})
                if isinstance(ids, dict):
                    arxiv_id = ids.get('arxiv')  # preferred explicit arXiv id

                # If arXiv id not present, try to parse from the pdf_url (common patterns)
                if not arxiv_id and arxiv_pdf_url:
                    # Common arXiv pdf URL patterns:
                    # https://arxiv.org/pdf/<id>.pdf or https://arxiv.org/abs/<id>
                    m = re.search(
                        r'arxiv\.org/(?:pdf|abs)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[A-Za-z\-]+/[0-9]{7}(?:v\d+)?)',
                        arxiv_pdf_url
                    )
                    if m:
                        arxiv_id = m.group(1)

                # Sanitize title for fallback filename usage
                raw_title = work.get('title') or work.get('display_name') or 'Untitled'
                safe_title = re.sub(r'[<>:"/\\|?*]', '', raw_title)[:100]

                # --- NEW: sanitize arXiv id for filename and ensure uniqueness ---
                filename: str
                if arxiv_id:
                    # Replace any characters that are invalid or problematic in filenames (including '/')
                    safe_arxiv_id = re.sub(r'[<>:"/\\|?*]', '_', arxiv_id)
                    # Truncate if extremely long (rare for arXiv ids)
                    safe_arxiv_id = safe_arxiv_id[:150]
                    filename = f"{safe_arxiv_id}.pdf"
                else:
                    filename = f"{safe_title}.pdf"

                # Keep arXiv id in the metadata (possibly None)
                paper_info: Dict[str, Any] = {
                    "id": work.get('id'),
                    "title": raw_title,
                    "filename": filename,
                    "pdf_url": arxiv_pdf_url,
                    "citations": work.get('cited_by_count'),
                    "arxiv_id": arxiv_id,
                }
                all_papers.append(paper_info)
                pbar.update(1)

            # Update cursor for next page
            params['cursor'] = data.get('meta', {}).get('next_cursor')
            if not params.get('cursor'):
                break

        except requests.exceptions.RequestException as e:
            print(f"\nError fetching metadata: {e}")
            break

    pbar.close()
    return all_papers

def _ensure_unique_filepath(save_dir: Path, desired_name: str) -> Path:
    """
    Ensure the filename is unique in save_dir by appending _1, _2, ... if needed.
    Returns a Path object for the unique file path.
    """
    candidate = save_dir / desired_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix or ".pdf"
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_candidate = save_dir / new_name
        if not new_candidate.exists():
            return new_candidate
        counter += 1

def download_papers(
    papers: List[Dict[str, Any]],
    output_dir: str,
    show_progress: bool = True
) -> None:
    """
    Downloads PDFs from the provided metadata list.
    Includes rate limiting to respect arXiv policies.
    """
    # Create directory if it doesn't exist
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ArXiv requires a custom User-Agent to avoid 403 Forbidden
    headers = {
        "User-Agent": "Mozilla/5.0 (Research Script; python-requests) contact: your_email@example.com"
    }

    iterator = tqdm(papers, desc="Downloading PDFs", disable=not show_progress, unit="file")

    for paper in iterator:
        # Use unique path to avoid overwriting files with same name
        desired_filename = paper['filename']
        file_path = _ensure_unique_filepath(save_path, desired_filename)

        # Skip if already exists (unique check above already ensures this won't overwrite)
        if file_path.exists():
            continue

        try:
            pdf_response = requests.get(paper['pdf_url'], headers=headers, stream=True, timeout=60)

            # Handle 403 or other errors
            if pdf_response.status_code != 200:
                iterator.write(f"Failed to download {paper['title']} (Status: {pdf_response.status_code})")
                continue

            with open(file_path, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Sleep to respect arXiv rate limits (avoid IP ban)
            time.sleep(3)

        except Exception as e:
            iterator.write(f"Error downloading {paper['title']}: {str(e)}")

def main() -> None:
    # Configuration
    N_PAPERS = 10_000
    OUTPUT_FOLDER = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/data/arxiv_pdfs"
    EMAIL_ADDRESS = "your_email@example.com"

    print(f"--- Starting Job: Top {N_PAPERS} ArXiv Papers ---")

    # 1. Get Metadata
    papers = get_top_arxiv_papers_metadata(
        n_papers=N_PAPERS,
        email=EMAIL_ADDRESS,
        show_progress=True
    )

    print(f"Found metadata for {len(papers)} papers. Starting download...")

    # 2. Download PDFs
    download_papers(
        papers=papers,
        output_dir=OUTPUT_FOLDER,
        show_progress=True
    )

    print(f"\nDone! Papers saved to {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()
