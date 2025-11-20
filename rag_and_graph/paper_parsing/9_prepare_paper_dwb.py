from d_markers_check import uses_numbered_citations
from sections_division import chunk_pdf_with_grobid
from map_refs_by_markers import match_refs_by_marker
from map_refs_by_names import get_matched_authors
from find_refs_by_grobid import find_refs
from dedupe_refs import deduplicate_refs
from ref_marker_tailing import parse_mark_refs
from locate_or_get_chunk_from_pdf import locate_chunk_in_pdf
# from dwm_meta_idx_quering import search_papers_parallel
from meta_idx_quering import ArxivMetaSearchDB, run_searches_multithreaded
from urllib.parse import urlparse
import tiktoken
from typing import Any, Dict, List, Optional
import hashlib
import os
import uuid
import re
import sys
import time
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Executor
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import pickle
from tqdm import tqdm
from typing import List, Dict, Any

# TODO
    # 1. Add tqdm for grobid pdf processing
    # 2. ???? Remake function that divides pdf into sections
    # 3. 

PG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "arxiv_meta_db_5",
    "user": "postgres",
    "password": "@Q_Fa;ml$f!@94r"
}
max_workers = 12
encoding_name = "cl100k_base"
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

def extract_arxiv_id(input_str: str) -> str:
    # Check if input is a URL
    if input_str.startswith("http://") or input_str.startswith("https://"):
        # Parse the URL to get the path part
        parsed_url = urlparse(input_str)
        # The arXiv ID is the last segment of the path, e.g. '1409.0473v7'
        arxiv_id = parsed_url.path.split('/')[-1]
        return arxiv_id
    else:
        # Assume input is a filepath, extract filename without extension
        filename = os.path.basename(input_str)
        arxiv_id = os.path.splitext(filename)[0]
        return arxiv_id

def locate_multiprocessed(sectins: dict):
    pass



pdf_paths = [
    "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1308.0850v5.pdf", # markers
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1508.04025v5.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1508.07909v5.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1511.06114v4.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1601.06733v7.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1602.02410v2.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1607.06450v1.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1608.05859v3.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1609.08144v2.pdf", # markers
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1610.02357v3.pdf", # markers
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1610.10099v2.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1701.06538v1.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1703.03130v1.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1703.10722v3.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1705.03122v3.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1705.04304v3.pdf",
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf", # markers
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1512.05287v5.pdf", # markers
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1602.01137v1.pdf", # markers
]

# GROBID with multithreading
def prepare_multithreaded(pdf_path: str) -> Any:
    sections = chunk_pdf_with_grobid(pdf_path)
    use_markers = uses_numbered_citations(pdf_path)
    seen_refs = set()
    arxiv_id = extract_arxiv_id(pdf_path)
    if use_markers:
        for secs in sections: # type: ignore
            refs = match_refs_by_marker(pdf_path, secs, as_list=True)['reference_mapping'] # secs =)))))
            seen_refs.update(refs)
        refs_tailed = parse_mark_refs(seen_refs) # type: ignore
        return sections, refs_tailed, use_markers, pdf_path, arxiv_id
    else:
        refs_content = find_refs(pdf_path)
        return sections, refs_content, use_markers, pdf_path, arxiv_id
start_time = time.time()
prepared_secs = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(prepare_multithreaded, pdf_path) for pdf_path in pdf_paths]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
        try:
            sections, refs, use_markers, pdf_path, arxiv_id = future.result()
            prepared_secs.append((sections, refs, use_markers, pdf_path, arxiv_id))
        except Exception as e:
            print(f"Task generated an exception: {e}")
end_time = time.time()
print(f"Total time taken for GROBIDing {len(pdf_paths)} pdfs: {end_time - start_time} seconds")




# Finalazing with multiprocessing
# Type aliases for clarity
Section = Dict[str, Any]
Ref = Dict[str, Any]
PreparedPaper = Tuple[List[Section], List[Ref], bool, str, str]  # (sections, refs, use_markers, pdf_path, arxiv_id)


def _is_picklable(obj: Any) -> bool:
    """Return True if obj can be pickled (simple test)."""
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def process_paper(paper: PreparedPaper, searcher: Any, max_workers: int, encoding_name: str) -> List[Dict[str, Any]]:
    """
    Worker that processes a single paper tuple and returns final_chunks (list of chunks).
    This function intentionally mirrors the logic you provided with minimal change.
    It must be top-level (module-level) so multiprocessing can import/pickle it.
    """
    sections, refs, use_markers, pdf_path, arxiv_id = paper
    section_found_refs: List[Tuple[str, str, Any]] = []
    final_chunks: List[Dict[str, Any]] = []

    print(sections[0])
    sys.exit()

    # numbered citation markers flow
    if use_markers:
        queries = searcher.prepare_queries(refs)
        cache_arxiv_found_refs: Dict[str, Any] = run_searches_multithreaded(searcher, queries, max_workers=max_workers)

        for sec in sections:
            section_found_refs = []
            sec_refs = match_refs_by_marker(pdf_path, sec, as_list=True)['reference_mapping']
            for ref in sec_refs:
                if ref in cache_arxiv_found_refs:
                    meta = cache_arxiv_found_refs[ref]['metadata']
                    item = (meta['id'], meta.get('title'), meta['authors']['author'])
                    if item not in section_found_refs:
                        section_found_refs.append(item)

            start, end = locate_chunk_in_pdf(sec['section_text'], pdf_path, n_first=6, n_last=6)
            final_chunks.append({
                "arxiv_id": arxiv_id,
                "chunk_id": str(uuid.uuid4()),
                "section_title": sec['section_title'],
                "chunk_text": sec['section_text'],
                "token_len": num_tokens_from_string(sec['section_text'], encoding_name),
                "is_citation_context": bool(section_found_refs),
                "citations": section_found_refs,
                "start_offset": start,
                "end_offset": end,
                "checksum": generate_checksum(sec['section_text']),
                "embedding_id": "weaviate_uuid_7f8a9c",
                "uses_markers": True
            })

    # author as citation markers flow
    else:
        refs_only: List[str] = [ref['raw'] for ref in refs]
        cleared_refs = set(deduplicate_refs(refs_only))
        filtered_refs = [ref for ref in refs if ref['raw'] in cleared_refs]
        queries = searcher.prepare_queries(filtered_refs)
        authors_only = [ref['authors_teiled'][0] for ref in refs]
        cache_arxiv_found_refs: Dict[str, Any] = run_searches_multithreaded(searcher, queries, max_workers=max_workers)  # expected dict

        for sec in sections:
            section_found_refs = []
            sec_authors = get_matched_authors(authors_only, sec['section_text'])
            sec_authors = searcher.normalize_author_list(sec_authors)

            for _, val in cache_arxiv_found_refs.items():
                # format_authors returns something indexable in original code
                formatted_authors = searcher.format_authors(val['metadata']['authors']['author'])
                if not formatted_authors:
                    continue
                searched_name = searcher.normalize_text(formatted_authors[0])
                if match_name(searched_name, sec_authors):
                    item = (val['metadata']['id'], val['metadata'].get('title'), val['metadata']['authors']['author'])
                    if item not in section_found_refs:
                        section_found_refs.append(item)

            start, end = locate_chunk_in_pdf(sec['section_text'], pdf_path, n_first=6, n_last=6)
            final_chunks.append({
                "arxiv_id": arxiv_id,
                "chunk_id": str(uuid.uuid4()),
                "section_title": sec['section_title'],
                "chunk_text": sec['section_text'],
                "token_len": num_tokens_from_string(sec['section_text'], encoding_name),
                "is_citation_context": bool(section_found_refs),
                "citations": section_found_refs,
                "start_offset": start,
                "end_offset": end,
                "checksum": generate_checksum(sec['section_text']),
                "embedding_id": "weaviate_uuid_7f8a9c",
                "uses_markers": False
            })

    return final_chunks


def parallel_process_papers(prepared_secs: Sequence[PreparedPaper], searcher: Any, 
                            max_workers: int = 8, encoding_name: str = "utf-8", prefer_processes: bool = True,) -> List[List[Dict[str, Any]]]:
    """
    Parallelizes processing across papers (top-level entries in prepared_secs).
    - If searcher is picklable and prefer_processes is True, uses ProcessPoolExecutor.
    - Otherwise uses ThreadPoolExecutor.
    Returns a list of final_chunks lists (same shape as original results list).
    """
    total = len(prepared_secs)
    if total == 0:
        return []

    # Decide executor type
    use_processes = prefer_processes and _is_picklable(searcher)

    executor_cls: Executor
    executor_label: str
    if use_processes:
        executor_cls = ProcessPoolExecutor # type: ignore
        executor_label = "processes (multiprocessing)"
    else:
        executor_cls = ThreadPoolExecutor # type: ignore
        executor_label = "threads"

    # Build an iterable of args to pass to worker. Each item is just the paper tuple:
    # process_paper accepts (paper, searcher, max_workers, encoding_name)
    # We use executor.map which preserves input order in results.
    map_args_iterable = ((paper, searcher, max_workers, encoding_name) for paper in prepared_secs)

    results: List[List[Dict[str, Any]]] = []
    # When using ProcessPoolExecutor.map with multiple args, use a tiny wrapper:
    def _map_wrapper(args_iter):
        # map a function that takes multiple args by using a lambda that unpacks
        return executor.map(lambda a: process_paper(*a), args_iter)

    # Execute with progress bar
    with executor_cls(max_workers=max_workers) as executor: # type: ignore
        # Executor.map with a generator of tuples: we can't directly pass generator to
        # executor.map with multiple args unless we use starmap-like behavior. Simpler:
        # create a list of tuples for stable behavior (size = number of papers)
        arg_list = list(map_args_iterable)
        # Use executor.map and wrap with tqdm to show progress
        # Here we pass each tuple as single argument to a small wrapper function that unpacks.
        # For ProcessPoolExecutor, the lambda won't be picklable. So instead use executor.map
        # against a small helper that is defined at top-level: process_paper unpacks args.
        # We'll use executor.map with starmap-like pattern by calling process_paper with each tuple:
        if use_processes:
            # ProcessPoolExecutor: use executor.map with a helper that accepts the whole tuple,
            # so ensure that worker callable is top-level. We'll use a small top-level helper below.
            def _worker_unpack(args_tuple: Tuple[PreparedPaper, Any, int, str]) -> List[Dict[str, Any]]:
                return process_paper(args_tuple[0], args_tuple[1], args_tuple[2], args_tuple[3])

            # map returns results in order
            for res in tqdm(executor.map(_worker_unpack, arg_list), total=total, desc=f"Processing papers ({executor_label})"):
                results.append(res)
        else:
            # ThreadPoolExecutor: lambdas are ok but keep consistent and use same _worker_unpack
            def _worker_unpack(args_tuple: Tuple[PreparedPaper, Any, int, str]) -> List[Dict[str, Any]]:
                return process_paper(args_tuple[0], args_tuple[1], args_tuple[2], args_tuple[3])

            for res in tqdm(executor.map(_worker_unpack, arg_list), total=total, desc=f"Processing papers ({executor_label})"):
                results.append(res)

    return results


start_time = time.time()

searcher = ArxivMetaSearchDB(PG, pool_minconn=1, pool_maxconn=20)
results = parallel_process_papers(prepared_secs, searcher, max_workers=8, encoding_name="cl100k_base")

end_time = time.time()
print(f"Total time taken for FINALazing {len(pdf_paths)} pdfs: {end_time - start_time} seconds")

for r in results:
    print(r)
    print('+'*120)






















# single_ready_to_embed_paper = {
#   "paper_id": "1234.56789",
#   "title": "Example Research Paper Title",
#   "authors": [
#     {"name": "Author One", "affiliation": "Institution A"},
#     {"name": "Author Two", "affiliation": "Institution B"}
#     ],
#   "abstract": "This is the abstract of the paper.",
#   "publication_year": 2024,
#   "chunks": [
#     {
#       "chunk_id": "chunk_1",
#       "text": "This is the first chunk of the paper text ...",
#       "embedding": [0.12, -0.34, 0.56, ..., 0.78],
#       "citations": [
#         {
#           "citation_text": "[1]",
#           "reference_id": "arxiv:9876.54321",
#           "context": "This chunk discusses related work from [1]..."
#         }
#       ]
#     },
#     {
#       "chunk_id": "chunk_2",
#       "text": "This is the second chunk of the paper text ...",
#       "embedding": [0.23, -0.45, 0.67, ..., 0.89],
#       "citations": [
#         {
#           "citation_text": "[2]",
#           "reference_id": "arxiv:1122.33445",
#           "context": "Further explanation referencing [2] in this section..."
#         }
#       ]
#     }
#   ],
#   "chunks_ok": [
#         "chunk_id": "ac99312f-0fea-4349-a701-7cf18beab5c7",
#         "section_title": "...",
#         "chunk_text": "...",
#         "token_len": 999,
#         "is_citation_context": True,
#         "citations": [
#             {"arxiv_id": "7777.7777", "title": "...", "authors": ["author name 3", "author name 2"]},
#             {"arxiv_id": "6666.6666", "title": "...", "authors": ["author name 3", "author name 2"]},
#             ],
#         "start_offset": 0,
#         "end_offset": 412,
#         "checksum": "070a64ea183a6872555ad89f9e43e89a132dbc10cea79ddbe4b72202fb76b313",
#         "embedding_id": "weaviate_uuid_7f8a9c"],
#   "references": [
#     {
#       "reference_id": "arxiv:9876.54321",
#       "title": "Referenced Paper One",
#       "authors": ["Author A", "Author B"],
#       "year": 2020,
#       "doi": "10.1234/exampledoi1"
#     },
#     {
#       "reference_id": "arxiv:1122.33445",
#       "title": "Referenced Paper Two",
#       "authors": ["Author C"],
#       "year": 2021,
#       "doi": "10.1234/exampledoi2"
#     }
#   ]
# }