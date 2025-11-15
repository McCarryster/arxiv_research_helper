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
import re
import sys
import time

# 1. Downnload pdf paper - 50%
# 2. Divide it in sections using GROBID
# 3. For each section - map each section with corresponding refs:
    # 1. Check if the sections has a) numbered markers as links to refs or b) author names as links to refs 
        # After a single bool check - use the same method for all sections. Only a) or b) method per pdf
        # a) If numbered markers as links to refs like "[13]", "[1, 2, 3]", "(13)", "(1, 2, 3)" -> use regex to find all refs
        # b) If author names as links to refs like "(Luong et al., 2015)" and other variants -> use GROBID to find all refs -> regex to dedup refs
    # 2. Map each section to corresponding reference from "references" section in pdf
        # If no refs -> "is_references": false -> skip
        # If numbered markers -> use regex to match section and refs
        # If author names -> use NER to match by names in section and names in refs
    # 3. Append found corresponding refs to the section
# 4. Complete and return the json 

PG = {
"host": "localhost",
"port": 5432,
"dbname": "arxiv_meta_db_5",
"user": "postgres",
"password": "@Q_Fa;ml$f!@94r"
}
searcher = ArxivMetaSearchDB(PG, pool_minconn=1, pool_maxconn=20)

# 0. Helper functions
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# def string_checksum_md5(input_string: str) -> str:
#     # Create MD5 hash object
#     md5_hash = hashlib.md5()
#     # Encode the input string to bytes, then hash it
#     md5_hash.update(input_string.encode('utf-8'))
#     # Return the hexadecimal digest of the hash
#     return md5_hash.hexdigest()
def generate_checksum(text):
    # Create a new sha256 hash object
    sha256_hash = hashlib.sha256()
    # Update the hash object with the encoded text (bytes)
    sha256_hash.update(text.encode('utf-8'))
    # Return the hexadecimal digest string
    return sha256_hash.hexdigest()

# def extract_arxiv_id(pdf_path: str) -> str:
#     # Extract the base filename without extension
#     filename = os.path.basename(pdf_path)
#     arxiv_id = os.path.splitext(filename)[0]
#     return arxiv_id

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

# 1. Download pdf paper
def download_pdf():
    pass


# doi CAN BE USED FOR GETTING ARXIV ID


# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1308.0850v5.pdf" # markers
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1508.04025v5.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1508.07909v5.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1511.06114v4.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1601.06733v7.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1602.02410v2.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1607.06450v1.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1608.05859v3.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1609.08144v2.pdf" # markers
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1610.02357v3.pdf" # markers
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1610.10099v2.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1701.06538v1.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1703.03130v1.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1703.10722v3.pdf"
pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1705.03122v3.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1705.04304v3.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf" # markers
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1512.05287v5.pdf" # markers
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1602.01137v1.pdf" # markers

start_time = time.time()

# 2. Divide pdf into sections
sections = parse_pdf_with_grobid(pdf_path)


# 3. For each section

use_markers = uses_numbered_citations(pdf_path) # 3.1. Check if the sections has a) numbered markers as links to refs or b) author names as links to refs 
candidate_limit = 100   # tune down for speed, up for recall
top_k = 1
use_authors = False  # title-only mode (fast)
max_workers = 12

cache_arxiv_found_refs = None
seen_refs = set()              # cache refs
# grobid_res = 
for i, section in enumerate(sections): # type: ignore
    section_found_refs = []
    # {"[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.": {'id': 772930, 'arxiv_id': '1412.3555', 'title': 'Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling', 'normalized_title': 'empirical evaluation of gated recurrent neural networks on sequence modeling', 'normalized_authors': ['bengio yoshua', 'cho kyunghyun', 'chung junyoung', 'gulcehre caglar'], 'title_and_authors': 'Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling | bengio yoshua, cho kyunghyun, chung junyoung, gulcehre caglar', 'created_date': datetime.date(2014, 12, 11)}}
    # print(section)
    print("#"*100)
    print(f"use_markers={use_markers}")
    print("#"*100)

    if use_markers: # If 3.1. a) If numbered markers as links to refs like "[13]", "[1, 2, 3]", "(13)", "(1, 2, 3)" -> use regex to find all refs
        # To iterate over all refs only once and cache
        if not seen_refs:
            for secs in sections: # type: ignore
                refs = match_refs_by_marker(pdf_path, secs, as_list=True)['reference_mapping'] # secs =)))))
                seen_refs.update(refs)
            refs_tailed = parse_mark_refs(seen_refs) # type: ignore
            queries = searcher.prepare_queries(refs_tailed)
            try:
                cache_arxiv_found_refs = run_searches_multithreaded(searcher, queries, max_workers=max_workers)
            finally:
                searcher.close()

        # To get values from cache and match section with refs
        if cache_arxiv_found_refs:
            sec_refs = match_refs_by_marker(pdf_path, section, as_list=True)['reference_mapping']
            for ref in sec_refs:
                if ref in cache_arxiv_found_refs:
                    section_found_refs.append(cache_arxiv_found_refs[ref])
        else:
            print('SOMETHING IS SHIT')
            break

        if not seen_refs:
            for section in sections: # type: ignore
                refs = match_refs_by_marker(pdf_path, section, as_list=True)
                refs = refs['reference_mapping']
                # refs = set(refs)
                seen_refs.update(refs)
            seen_refs = {s for s in seen_refs if "Reference not found" not in s}
            refs_tailed = parse_mark_refs(seen_refs) # type: ignore
            for ref in refs_tailed:
                print(f"{ref},")
            break
            parallel_results = search_papers_parallel(refs_tailed, db_path, top_k=top_k, use_authors=use_authors, max_workers=max_workers)
                # cache_arxiv_found_refs[]
            # print(parallel_results[0])
            for i, p_res in enumerate(parallel_results):
                print(i+1, p_res[0]['arxiv_id'], p_res[0]['title'], p_res[0]['authors']) # type: ignore
                print('-'*100)
        break

        refs = match_refs_by_marker(pdf_path, section, as_list=True)




        # for seen_ref in filtered_set:
        #     print(seen_ref)
        #     print('-'*100)
        # break
        # for i, ref in enumerate(refs_tailed):
        #     print(f"{ref},")
        print('^'*100)
        # print(parallel_results)
        for i, res in enumerate(parallel_results):
            if res:
                print(i+1, res[0]['arxiv_id'], res[0]['title'], res[0]['authors']) # type: ignore
            else:
                print(i, "[NOT FOUND]")
            print("-"*100)
        
            # tailed_ref = parse_mark_ref(ref)
            # print(f"{ref},")
            # print({"authors_teiled": tailed_ref["authors_teiled"], "title": tailed_ref["title"]})
            # if ref not in cache_arxiv_found_refs:
            # matched_refs.append(ref)
        # MAKE A GROBID TO A REF STRING TO MAKE IT STRUCTURED

    else:           # If 3.1. b) If author names as links to refs like "(Luong et al., 2015)" and other variants -> use GROBID to find all refs
        
        if not cache_arxiv_found_refs:
        
            refs_content = find_refs(pdf_path)                                    # grobid parse to find of references
            refs_only = [ref['raw'] for ref in refs_content]                      # take only raw references from grobid content
        authors_only = [ref['authors_teiled'][0] for ref in refs_content]         # take only authors from grobid content
        print()
        section_authors = get_matched_authors(authors_only, section['text'])      # find section
        section_authors = searcher.normalize_author_list(section_authors)
        print("#"*100)
        print("section_authors", section_authors)
        print("#"*100)
        cleared_refs = deduplicate_refs(refs_only)
        cleared_refs_set = set(cleared_refs)
        filtered_refs = [ref for ref in refs_content if ref['raw'] in cleared_refs_set]
        
        queries = searcher.prepare_queries(filtered_refs)
        try:
            cache_arxiv_found_refs = run_searches_multithreaded(searcher, queries, max_workers=max_workers)
        finally:
            searcher.close()
        
        for key, val in cache_arxiv_found_refs.items():
            print(key, "|||", val['normalized_authors'])
            print('-'*100)
        





        # for i, ref in enumerate(cache_arxiv_found_refs):
        #     # -------------------------------REMOVE
        #     # checks = {'title': ref['title'], 'authors_teiled': ref['authors_teiled']}
        #     # print(ref)
        #     # print(f"`{checks}`,")
        #     print(ref)
        #     # break
        #     # -------------------------------REMOVE



            # resp = find_arxiv(ref)
            # # print(resp)
            # if resp:
            #     cache_arxiv_found_refs[ref['raw']] = (extract_arxiv_id(resp.get('id')), resp.get('title')) # type: ignore
            # else:
            #     cache_arxiv_found_refs[ref['raw']] = (None, ref['raw'])
            # # print(cache_arxiv_found_refs)
            # if ref['authors_teiled'][0] in section_authors:
            #     # print("g+", i+1, ref['authors_teiled'][0], "|", ref['raw'])
            #     # print("-"*100)
            #     matched_refs.append(ref['raw'])

    break

end_time = time.time()
print(f"Total time taken for the loop: {end_time - start_time} seconds")
    # break

    # print("matched refs:", len(matched_refs))
    # for match_ref in matched_refs:
    #     if match_ref in cache_arxiv_found_refs:
    #         citations.append({
    #             'arxiv_id': cache_arxiv_found_refs[match_ref][0],
    #             'title': cache_arxiv_found_refs[match_ref][1],
    #             'raw_reference': match_ref
    #         })
    # chunk_json['chunk_id'] = extract_arxiv_id(pdf_path)
    # chunk_json['text'] = section['text']
    # chunk_json['start_offset'] = None
    # chunk_json['end_offset'] = None
    # chunk_json['section'] = section['section'].lower()
    # chunk_json['token_len'] = num_tokens_from_string(section['text'], "cl100k_base")
    # chunk_json['embedding_id'] = 9999999
    # if citations:
    #     chunk_json['is_citation_context'] = True
    #     chunk_json['citations'] = citations
    # chunk_json['checksum'] = generate_checksum(section['text'])


    # print(chunk_json)
    # break