from map_refs_by_names import get_matched_authors
from dedupe_refs import deduplicate_refs
from meta_idx_quering import ArxivMetaSearchDB, run_searches_multithreaded
from map_refs_by_markers import match_refs_by_marker
from grobid_processing_part import grobid_processing
from config import *

import tiktoken
import os
import hashlib
import uuid
import json
import sys
import time
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from multiprocessing import Pool, current_process
from tqdm import tqdm
import logging

logging.basicConfig(filename=log_dir, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# UPDATES NEEDED:
    # 1. Add citation_count: int to every paper
    # 2. Make saving as JSONL
    # 3. Add str_authors field {"keyname": "Marwan", "forenames": "N." }, -> ["Marwan N."] for everything
    # 4. Remake chunk citations from list to dicts
    # 5. Rename embedding_id to chunk_embedding_id
    # 6. Add title_abstract_embedding_id field to final paper

# --- Type Definitions ---
Sections = List[Dict[str, Any]]
Ref = List[Dict[str, Any]]
PreparedPaper = Tuple[Sections, Ref, bool, str, str]  # (sections, refs, use_markers, pdf_path, arxiv_id)
ProcessedResult = Dict[str, Any]

# 0. Helper functions
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def match_name(name: str, name_list: List[str]) -> bool:
    """Match a name with a list of names ignoring order of first and last names."""
    # Normalize the input name (lowercase and split)
    name_parts = set(name.lower().split())

    # Check against each name in the list
    for candidate in name_list:
        candidate_parts = set(candidate.lower().split())
        if name_parts == candidate_parts:
            return True
    return False

def generate_checksum(text):
    # Create a new sha256 hash object
    sha256_hash = hashlib.sha256()
    # Update the hash object with the encoded text (bytes)
    sha256_hash.update(text.encode('utf-8'))
    # Return the hexadecimal digest string
    return sha256_hash.hexdigest()

# --- Worker State Management ---

# This global variable will exist only within the worker process scope
worker_searcher: Optional[ArxivMetaSearchDB] = None

def init_worker(pg_config: Dict[str, Any], thread_pool_size: int) -> None:
    """
    Initializes the database connection for each worker process.
    
    Args:
        pg_config: The Postgres configuration dictionary.
        thread_pool_size: How many threads this specific process is allowed to use 
                          internally for DB calls.
    """
    global worker_searcher
    try:
        # We create a new DB instance per process. 
        # pool_maxconn needs to match the internal threading logic (max_workers)
        # so that threads don't starve waiting for a DB connection.
        worker_searcher = ArxivMetaSearchDB(
            pg_config, 
            pool_minconn=1, 
            pool_maxconn=thread_pool_size
        )
    except Exception as e:
        print(f"Error initializing worker {current_process().name}: {e}")
        sys.exit(1)

# --- Core Logic ---

def process_paper(paper: PreparedPaper) -> Dict[str, Any]:
    """
    The main logic function. It now uses the process-local 'worker_searcher' 
    instead of a global object.
    """
    if worker_searcher is None:
        raise RuntimeError("Worker searcher not initialized. check init_worker.")
    
    sections, refs, use_markers, pdf_path, arxiv_id = paper
    arxiv_id = arxiv_id.replace("_", "/")
    
    # 1. Fetch main paper metadata
    main_paper_meta = worker_searcher.search_paper_by_arxiv_id(arxiv_id)
    if not main_paper_meta:
        logging.info(f'BAD! {arxiv_id} paper was not found in arxiv_meta_db')
        return {}

    # section_found_refs: List[Tuple[str, str, Any, Any, int]] = []
    section_found_refs: List[Dict[str, Any]] = []
    final_chunks: List[Dict[str, Any]] = []

    # 2. Numbered citation markers flow
    if use_markers:
        queries = worker_searcher.prepare_queries(refs)
        # Note: max_workers here refers to threads *inside* this single process
        cache_arxiv_found_refs: Dict[str, Any] = run_searches_multithreaded(worker_searcher, queries, max_workers=max_workers)

        for i, sec in enumerate(sections):
            section_found_refs = []
            sec_refs = match_refs_by_marker(pdf_path, sec, as_list=True)['reference_mapping']

            for ref in sec_refs:
                if ref in cache_arxiv_found_refs:
                    meta = cache_arxiv_found_refs[ref]['metadata']
                    citation_count = cache_arxiv_found_refs[ref].get('citation_count', None)
                    item = {
                        "arxiv_id": meta['id'],
                        "title": meta.get('title'),
                        "authors": meta['authors']['author'],
                        "authors_str": worker_searcher.format_authors(meta['authors']['author']),
                        "citation_count": citation_count
                    }
                    if item not in section_found_refs:
                        section_found_refs.append(item)

            final_chunks.append({
                "chunk_num": i+1,
                "chunk_id": str(uuid.uuid4()),
                "section_title": sec['section_title'],
                "chunk_text": sec['section_text'],
                "token_len": num_tokens_from_string(sec['section_text'], encoding_name),
                "is_citation_context": bool(section_found_refs),
                "citations": section_found_refs,
                "checksum": generate_checksum(sec['section_text']),
                "uses_markers": True,
                "embedding_id": ""
            })

    # 3. Author as citation markers flow
    else:
        refs_only: List[str] = [ref['raw'] for ref in refs]
        cleared_refs = set(deduplicate_refs(refs_only))
        filtered_refs = [ref for ref in refs if ref['raw'] in cleared_refs]
        
        queries = worker_searcher.prepare_queries(filtered_refs)
        # authors_only = [ref['authors_teiled'][0] for ref in refs] # FIX
        authors_only = [ref['authors_teiled'][0] for ref in refs if ref.get('authors_teiled')]
        
        # Multithreaded search using the local worker_searcher
        cache_arxiv_found_refs: Dict[str, Any] = run_searches_multithreaded(worker_searcher, queries, max_workers=max_workers)

        for i, sec in enumerate(sections):
            section_found_refs = []
            sec_authors = get_matched_authors(authors_only, sec['section_text'])
            sec_authors = worker_searcher.normalize_author_list(sec_authors)

            for _, val in cache_arxiv_found_refs.items():
                formatted_authors = worker_searcher.format_authors(val['metadata']['authors']['author'])
                citation_count = val.get('citation_count', None)
                if not formatted_authors:
                    continue
                searched_name = worker_searcher.normalize_text(formatted_authors[0])
                if match_name(searched_name, sec_authors):
                    item = {
                        "arxiv_id": val['metadata']['id'],
                        "title": val['metadata'].get('title'),
                        "authors": val['metadata']['authors']['author'],
                        "authors_str": worker_searcher.format_authors(val['metadata']['authors']['author']),
                        "citation_count": citation_count
                    }
                    if item not in section_found_refs:
                        section_found_refs.append(item)

            final_chunks.append({
                "chunk_num": i+1,
                "chunk_id": str(uuid.uuid4()),
                "section_title": sec['section_title'],
                "chunk_text": sec['section_text'],
                "token_len": num_tokens_from_string(sec['section_text'], encoding_name),
                "is_citation_context": bool(section_found_refs),
                "citations": section_found_refs,
                "checksum": generate_checksum(sec['section_text']),
                "uses_markers": False,
                "chunk_embedding_id": ""
            })

    # 4. Gather overall references
    overall_refs: List[Dict[str, Any]] = []
    for _, meta in cache_arxiv_found_refs.items():
        overall_refs.append({
            'arxiv_id': meta['arxiv_id'], 
            'title': meta['metadata']['title'],
            'citation_count': meta.get('citation_count', None),
            'authors': meta['metadata']['authors']['author'],
            'authors_str': worker_searcher.format_authors(meta['metadata']['authors']['author']),
        })

    # 5. Final Construct
    single_ready_to_embed_paper = {
        "arxiv_id": arxiv_id,
        "title": main_paper_meta['title'],
        "authors": main_paper_meta['metadata']['authors']['author'],
        "authors_str": worker_searcher.format_authors(main_paper_meta['metadata']['authors']['author']),
        "abstract": main_paper_meta['metadata']['abstract'],
        "created": main_paper_meta['metadata']['created'],
        "updated": main_paper_meta['metadata'].get('updated', ''),
        "chunks": final_chunks,
        "references": overall_refs,
        "citation_count": main_paper_meta.get('citation_count', None),
        "paper_embedding_id": "",
    }
    return single_ready_to_embed_paper

def process_paper_safe(paper: PreparedPaper) -> Optional[Dict[str, Any]]:
    """
    Wrapper to catch exceptions in subprocesses so the pool doesn't hang.
    """
    try:
        return process_paper(paper)
    except Exception as e:
        # Log the error with the arxiv_id if available
        arxiv_id = paper[4] if len(paper) > 4 else "Unknown"
        logging.info(f"\n[Error] Failed processing {arxiv_id}: {e}")
        traceback.print_exc()
        return None

# --- Main Execution ---
# def run_processing_pipeline(papers_data: List[PreparedPaper], show_progress: bool = True, num_processes: int = 4) -> List[ProcessedResult]:
def run_processing_pipeline(papers_data: List[PreparedPaper], final_papers_path: str, show_progress: bool = True, num_processes: int = 4) -> Set[str]:
    """
    Process papers in parallel and persist intermediate results to allow resuming.

    Parameters
    ----------
    papers_data:
        List of PreparedPaper tuples: (sections, refs, use_markers, pdf_path, arxiv_id)
    show_progress:
        Whether to show a tqdm progress bar for overall progress.
    num_processes:
        Number of worker processes for multiprocessing.Pool.

    Returns
    -------
    Set[str]
        Set of processed ids (new + already processed, if any)
    """
    
    # processed_results: List[ProcessedResult] = []
    processed_ids: Set[str] = set()
    if os.path.exists(final_papers_path):
        print(f"📂 Scanning existing file: {final_papers_path}...")
        try:
            with open(final_papers_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            if 'id' in record:
                                processed_ids.add(record['arxiv_id'])
                        except json.JSONDecodeError as e:
                            print(f"Decode error {e}: {line}")
                            continue
            print(f"✅ Resuming: Found {len(processed_ids)} processed papers.")
        except Exception as e:
            print(f"⚠️ Error reading existing data: {e}. Starting fresh.")


    # Filter out already-processed papers
    papers_to_process: List[PreparedPaper] = [p for p in papers_data if (len(p) >= 5 and str(p[4]) not in processed_ids)]

    if not papers_to_process:
        return set() # Nothing new to do

    pool_args = (PG, max_workers)
    # Create iterator from pool
    with open(final_papers_path, 'a', encoding='utf-8', buffering=1) as f_out:
        with Pool(processes=num_processes, initializer=init_worker, initargs=pool_args) as pool:
            iterator = pool.imap_unordered(process_paper_safe, papers_to_process)

            if show_progress:
                iterator = tqdm(iterator, total=len(papers_to_process), desc="Processing Papers")

            for result in iterator:
                if result:
                    arxiv_id = str(result.get("arxiv_id")) if isinstance(result, dict) else None
                    if arxiv_id and arxiv_id not in processed_ids:
                        processed_ids.add(arxiv_id)
                        f_out.write(json.dumps(result) + "\n")

    return processed_ids

if __name__ == "__main__":
    start_time = time.time()

    # GROBID call (once)
    # grobid_processing()

    with open(grobid_prepared_path, 'r') as f:
        prepared_papers = json.load(f)

    try:
        if not prepared_papers:
            print("No papers to process (input_papers list is empty).")
        else:
            results = run_processing_pipeline(prepared_papers, final_papers_path, show_progress=True, num_processes=4)
            
            end_time = time.time()
            print(f"SUCCESS!!! TOOK FOR ALL SHIT {len(results)} pdfs: {end_time - start_time} seconds")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(0)