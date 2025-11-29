import os
import json
import re
import time
import requests
from typing import List, Dict, Any, Optional, Set
from tqdm.auto import tqdm

def get_top_arxiv_papers_metadata(
    n_papers: int = 200,
    email: str = "your_email@example.com",
    show_progress: bool = True,
    save_meta_path: str = "data/arxiv_citation_counts.jsonl", # Changed to .jsonl
    cursor_save_path: str = "data/arxiv_next_cursor.txt",
) -> None: # Returns None because data is too large to return in a list
    """
    Fetches metadata for the top N most cited papers that have an arXiv version.
    
    Optimized for massive datasets (millions of records): 
    - Uses JSONL (Append mode) instead of JSON List to prevent I/O bottlenecks.
    - Uses Set for O(1) deduplication lookup.
    - Does not hold full data in RAM to prevent MemoryErrors.
    """
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_meta_path)), exist_ok=True)

    base_url = "https://api.openalex.org/works"
    arxiv_source_id = "S4306400194"
    
    # --- STATE TRACKING ---
    seen_ids: Set[str] = set()
    papers_collected_count = 0
    start_cursor = "*"

    # 1. Load existing state (Count lines and load IDs into set)
    if os.path.exists(save_meta_path):
        print(f"📂 Scanning existing file: {save_meta_path}...")
        try:
            with open(save_meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            if 'id' in record:
                                seen_ids.add(record['id'])
                                papers_collected_count += 1
                        except json.JSONDecodeError:
                            continue
            print(f"✅ Resuming: Found {papers_collected_count} existing papers.")
        except Exception as e:
            print(f"⚠️  Error reading existing data: {e}. Starting fresh.")

    # 2. Load cursor
    if os.path.exists(cursor_save_path):
        try:
            with open(cursor_save_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    start_cursor = content
            print(f"✅ Resuming from cursor: {start_cursor[:50]}...")
        except Exception as e:
            print(f"⚠️  Could not read cursor: {e}. Using default '*'")

    # Early exit
    if papers_collected_count >= n_papers:
        print(f"✅ Target reached ({papers_collected_count} >= {n_papers}). Done!")
        return

    # Progress bar
    pbar = tqdm(total=n_papers, initial=papers_collected_count, 
               desc="Fetching Papers", 
               disable=not show_progress, unit="paper")

    # API Parameters
    params = {
        "filter": f"locations.source.id:{arxiv_source_id}",
        "sort": "cited_by_count:desc",
        "per-page": 200,
        "mailto": email,
        "cursor": start_cursor
    }

    # Open file in APPEND mode ('a')
    # buffering=1 means line buffering (writes to disk roughly every line)
    with open(save_meta_path, 'a', encoding='utf-8', buffering=1) as f_out:
        
        while papers_collected_count < n_papers:
            try:
                # Polite delay to avoid rate limits
                time.sleep(0.2) 
                
                response = requests.get(base_url, params=params, timeout=60)
                
                if response.status_code == 429:
                    print("\n⏳ Rate limit hit. Sleeping 10s...")
                    time.sleep(10)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                results = data.get('results', [])
                if not results:
                    print("\n❌ No more results available from API.")
                    break

                batch_papers: List[Dict[str, Any]] = []

                for work in results:
                    if papers_collected_count >= n_papers:
                        break

                    # ID Check (O(1) lookup)
                    work_id = work.get('id')
                    if work_id in seen_ids:
                        continue

                    # Logic: Extract arXiv info
                    arxiv_pdf_url: Optional[str] = None
                    for location in work.get('locations', []):
                        src = location.get('source')
                        if src and isinstance(src, dict) and src.get('id') and str(src.get('id')).endswith(arxiv_source_id):
                            arxiv_pdf_url = location.get('pdf_url')
                            break

                    if not arxiv_pdf_url:
                        continue

                    # Logic: Extract arXiv ID
                    arxiv_id: Optional[str] = None
                    ids = work.get('ids', {})
                    if isinstance(ids, dict):
                        arxiv_id = ids.get('arxiv')

                    if not arxiv_id and arxiv_pdf_url:
                        # Fallback regex
                        m = re.search(
                            r'arxiv\.org/(?:pdf|abs)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[A-Za-z\-]+/[0-9]{7}(?:v\d+)?)',
                            arxiv_pdf_url
                        )
                        if m:
                            arxiv_id = m.group(1)

                    # Logic: Filename
                    raw_title = work.get('title') or work.get('display_name') or 'Untitled'
                    safe_title = re.sub(r'[<>:"/\\|?*]', '', raw_title)[:100]
                    
                    if arxiv_id:
                        safe_arxiv_id = re.sub(r'[<>:"/\\|?*]', '_', arxiv_id)[:150]
                        filename = f"{safe_arxiv_id}.pdf"
                    else:
                        filename = f"{safe_title}.pdf"

                    paper_info: Dict[str, Any] = {
                        "id": work_id,
                        "title": raw_title,
                        "filename": filename,
                        "pdf_url": arxiv_pdf_url,
                        "citations": work.get('cited_by_count'),
                        "arxiv_id": arxiv_id,
                    }
                    
                    # Add to batch and memory set
                    batch_papers.append(paper_info)
                    seen_ids.add(work_id)
                    papers_collected_count += 1
                    pbar.update(1)

                # Write batch to JSONL file immediately
                if batch_papers:
                    for p in batch_papers:
                        f_out.write(json.dumps(p) + "\n")
                
                # Update Cursor
                next_cursor = data.get('meta', {}).get('next_cursor')
                if next_cursor:
                    params['cursor'] = next_cursor
                    # Save cursor to file
                    with open(cursor_save_path, 'w', encoding='utf-8') as f_curs:
                        f_curs.write(next_cursor)
                else:
                    print("\n🏁 Reached end of results (no next_cursor)")
                    break

            except requests.exceptions.RequestException as e:
                print(f"\n❌ Network Error: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
                # Don't break, retry loop

    pbar.close()
    
    # Cleanup cursor on full success
    if papers_collected_count >= n_papers and os.path.exists(cursor_save_path):
        os.remove(cursor_save_path)
        print("🗑️  Cleared cursor file (complete!)")
    
    print(f"🎉 COMPLETE: {papers_collected_count} papers saved to {save_meta_path}")

# --- EXECUTION ---

# Update paths to your local preferences
N_PAPERS = 3_000_000
EMAIL_ADDRESS = "michinemi964@gmail.com"
BASE_PATH = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/data"

get_top_arxiv_papers_metadata(
    n_papers=N_PAPERS,
    email=EMAIL_ADDRESS,
    show_progress=True,
    save_meta_path=f"{BASE_PATH}/arxiv_citation_counts.jsonl",
    cursor_save_path=f"{BASE_PATH}/arxiv_next_cursor.txt",
)