from d_markers_check import uses_numbered_citations
from sections_division import parse_pdf_with_grobid
from map_refs_by_markers import match_refs_by_marker
from map_refs_by_names import get_matched_authors
from find_refs_by_grobid import find_refs
from dedupe_refs import deduplicate_refs
from ref_marker_tailing import parse_mark_refs
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


pdf_paths = [
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1308.0850v5.pdf", # markers
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
    "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf", # markers
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1512.05287v5.pdf", # markers
    # "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1602.01137v1.pdf", # markers
]

# GROBID USE
    # parse_pdf_with_grobid(pdf_path)
    # When use_markers is True:
        # parse_mark_refs(seen_refs)
    # When use_markers is False:
        # find_refs(pdf_path)

# Multi threading steps:
    # 1. Pass 1 pdf to each worker
    # 2. Each worker should classify use_markers (True/False) for each
    # 3. Each worker should divide pdf into sections using parse_pdf_with_grobid()
    # 4. For each section each worker should
        # If use_markers is True:
            # Each worker should parse_mark_refs(seen_refs) for each pdf
        # If use_markers is False:
            # Each worker should find_refs(pdf_path) for each


# Input for workers:
    # 1. pdf_path
    # 2. searcher (a multithreaded postgreSQL search)


# Only GROBID I/O
# def prepare_multithreaded(pdf_paths: List: [str]):
    # 1. Return sections in a List of dicts
    # 2. Return use_markers (True/False) using uses_numbered_citations()
    # 3. Return refs_content using find_refs(pdf_path) if use_markers is True OR Return refs_tailed using parse_mark_refs(seen_refs)

def prepare_multithreaded(pdf_path: str) -> Any:
    sections = parse_pdf_with_grobid(pdf_path)
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
# with ThreadPoolExecutor(max_workers=10) as executor:
#     futures = [executor.submit(prepare_multithreaded, pdf_path) for pdf_path in pdf_paths]
#     for future in as_completed(futures):
#         try:
#             sections, refs, use_markers, pdf_path, arxiv_id = future.result()
#             prepared_secs.append((sections, refs, use_markers, pdf_path, arxiv_id))
#         except Exception as e:
#             print(f"Task generated an exception: {e}")
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(prepare_multithreaded, pdf_path) for pdf_path in pdf_paths]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
        try:
            sections, refs, use_markers, pdf_path, arxiv_id = future.result()
            prepared_secs.append((sections, refs, use_markers, pdf_path, arxiv_id))
        except Exception as e:
            print(f"Task generated an exception: {e}")
end_time = time.time()
print(f"Total time taken for {len(pdf_paths)} pdfs: {end_time - start_time} seconds")

searcher = ArxivMetaSearchDB(PG, pool_minconn=1, pool_maxconn=20)
# for sections, refs, use_markers in prepared_secs:
#     if use_markers:
#         print(use_markers)
#         print('#'*120)
#         print(sections[0])
#         print('#'*120)
#         print(refs[0])
#         print('#'*120)
#         break



example = {
    "9999.9999": {
        "arxiv_id": "9999.9999",
        "chunk_id": "ac99312f-0fea-4349-a701-7cf18beab5c7",
        "section_name": "...",
        "chunk_text": "...",
        "token_len": 999,
        "is_citation_context": None,
        "citations": [
            {"arxiv_id": "7777.7777", "title": "...", "authors": ["author name 3", "author name 2"]},
            {"arxiv_id": "6666.6666", "title": "...", "authors": ["author name 3", "author name 2"]}
            ],
        "start_offset": 0,
        "end_offset": 412,
        "checksum": "070a64ea183a6872555ad89f9e43e89a132dbc10cea79ddbe4b72202fb76b313",
        "embedding_id": "weaviate_uuid_7f8a9c"
    }
}
            # final.append({
            # arxiv_id: {
            #     "arxiv_id": arxiv_id,
            #     "chunk_id": str(uuid.uuid4()),
            #     "section_name": sec['section'],
            #     "chunk_text": sec['text'],
            #     "token_len": num_tokens_from_string(sec['text'], encoding_name),
            #     "is_citation_context": None,
            #     "citations": section_found_refs,
            #     "start_offset": 0,
            #     "end_offset": 412,
            #     "checksum": generate_checksum(sec['text']),
            #     "embedding_id": "weaviate_uuid_7f8a9c"
            # }})
start_time = time.time()
results = []
# 1 paper level
for sections, refs, use_markers, pdf_path, arxiv_id in prepared_secs:
    section_found_refs = []
    final_chunks = []
    # numbered citations
    if use_markers:
        print('wtf', use_markers)
        queries = searcher.prepare_queries(refs)
        cache_arxiv_found_refs = run_searches_multithreaded(searcher, queries, max_workers=max_workers) # dict
        # 1 section from 1 paper level
        for sec in sections:
            sec_refs = match_refs_by_marker(pdf_path, sec, as_list=True)['reference_mapping']
            for ref in sec_refs:
                if ref in cache_arxiv_found_refs:
                    item = (cache_arxiv_found_refs[ref]['metadata']['id'], cache_arxiv_found_refs[ref]['metadata']['title'], cache_arxiv_found_refs[ref]['metadata']['authors']['author'])
                    if item not in section_found_refs:
                        section_found_refs.append(item)
            # for sfr in section_found_refs:
            #     print(sec['section'])
            #     print(sfr)
            #     print('&'*120)

            final_chunks.append({
                "arxiv_id": arxiv_id,
                "chunk_id": str(uuid.uuid4()),
                "section_name": sec['section'],
                "chunk_text": sec['text'],
                "token_len": num_tokens_from_string(sec['text'], encoding_name),
                "is_citation_context": bool(section_found_refs),
                "citations": section_found_refs,
                "start_offset": 0,
                "end_offset": 412,
                "checksum": generate_checksum(sec['text']),
                "embedding_id": "weaviate_uuid_7f8a9c",
                "uses_markers": True
            })
            section_found_refs = []

    # author names as citations
    else:
        print('wtf', use_markers)
        refs_only = [ref['raw'] for ref in refs]
        cleared_refs = set(deduplicate_refs(refs_only))
        filtered_refs = [ref for ref in refs if ref['raw'] in cleared_refs]
        queries = searcher.prepare_queries(filtered_refs)
        authors_only = [ref['authors_teiled'][0] for ref in refs]
        cache_arxiv_found_refs = run_searches_multithreaded(searcher, queries, max_workers=max_workers) # dict
        # 1 section from 1 paper level
        for sec in sections:
            sec_authors = get_matched_authors(authors_only, sec['text'])
            sec_authors = searcher.normalize_author_list(sec_authors)

            for _, val in cache_arxiv_found_refs.items(): # raw (not used), meta
                searched_name = searcher.normalize_text(searcher.format_authors(val['metadata']['authors']['author'])[0])
                if match_name(searched_name, sec_authors):
                    item = (val['metadata']['id'], val['metadata']['title'], val['metadata']['authors']['author'])
                    if item not in section_found_refs:
                        section_found_refs.append(item)
                        # print(item)
                        # print('-'*100)

            final_chunks.append({
                "arxiv_id": arxiv_id,
                "chunk_id": str(uuid.uuid4()),
                "section_name": sec['section'],
                "chunk_text": sec['text'],
                "token_len": num_tokens_from_string(sec['text'], encoding_name),
                "is_citation_context": bool(section_found_refs),
                "citations": section_found_refs,
                "start_offset": 0,
                "end_offset": 412,
                "checksum": generate_checksum(sec['text']),
                "embedding_id": "weaviate_uuid_7f8a9c",
                "uses_markers": False
            })
            section_found_refs = []

    results.append(final_chunks)

end_time = time.time()
print(f"Total time taken for preparing {len(pdf_paths)} pdfs: {end_time - start_time} seconds")

# lens = []
# for f in results:
#     # if not f['uses_markers']:
#         # print(f['arxiv_id'], f['token_len'])
#         # print()
#         # print(f['section_name'])
#         # print()
#         # print(len(f['citations']), f['citations'])
#         # print("'"*120)
#         # lens.append(len(f['citations']))
#         print(f)
#         print(len(f['citations']))
#         print("'"*120)
# print(lens)

for r in results:
    print(r)
    print('+'*120)

print(len(results))






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
#         "section_name": "...",
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