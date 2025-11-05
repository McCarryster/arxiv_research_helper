from bool_check import check_json_for_markers
from sections_division import parse_pdf_with_grobid
from map_refs_by_markers import match_refs_by_marker
from map_refs_by_names import get_matched_authors
from find_refs_by_grobid import find_refs
from dedupe_refs import deduplicate_refs
from arxiv_paper_search import find_arxiv
from urllib.parse import urlparse
import tiktoken
import hashlib
import os
import re

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
pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1508.04025v5.pdf"
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
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1705.03122v3.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1705.04304v3.pdf"
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf" # markers
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1512.05287v5.pdf" # markers
# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1602.01137v1.pdf" # markers

# 2. Divide pdf into sections
sections = parse_pdf_with_grobid(pdf_path)


# 3. For each section
use_markers=False
for i, section in enumerate(sections): # type: ignore
    matched_refs = []
    cache_arxiv_found_refs = {}
    chunk_json = {}
    citations = []
    print(section)
    bool_check = check_json_for_markers(section) # 3.1. Check if the sections has a) numbered markers as links to refs or b) author names as links to refs 
    if bool_check:
        use_markers=True

    print("-"*100)
    print(f"use_markers={use_markers}, bool_check={bool_check}")
    print("-"*100)

    

    if use_markers: # If 3.1. a) If numbered markers as links to refs like "[13]", "[1, 2, 3]", "(13)", "(1, 2, 3)" -> use regex to find all refs
        mapping = match_refs_by_marker(pdf_path, section, as_list=True)
        mapping = mapping['reference_mapping']
        for i, ref in enumerate(mapping):
            # print("m+", i+1, ref)
            # print("-"*100)
            # if ref not in cache_arxiv_found_refs:
            matched_refs.append(ref)
        # MAKE A GROBID TO A REF STRING TO MAKE IT STRUCTURED

    else:           # If 3.1. b) If author names as links to refs like "(Luong et al., 2015)" and other variants -> use GROBID to find all refs
        refs_content = find_refs(pdf_path)
        print("-"*100)
        refs_only = [ref['raw'] for ref in refs_content]
        authors_only = [ref['authors_teiled'][0] for ref in refs_content]
        section_authors = get_matched_authors(authors_only, section['text'])
        print("#"*100)
        print("section_authors", section_authors)
        print("#"*100)
        cleared_refs = deduplicate_refs(refs_only)
        cleared_refs_set = set(cleared_refs)
        filtered_refs = [ref for ref in refs_content if ref['raw'] in cleared_refs_set]
        
        for i, ref in enumerate(filtered_refs):


            # -------------------------------REMOVE
            # if i == 2:
            #     print(ref)
            #     break
            # continue
            checks = {'title': ref['title'], 'authors_teiled': ref['authors_teiled']}
            print(f"{checks},")
            # print('-'*100)
            if i == 15:
                break
            # -------------------------------REMOVE

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

    print("matched refs:", len(matched_refs))
    for match_ref in matched_refs:
        if match_ref in cache_arxiv_found_refs:
            citations.append({
                'arxiv_id': cache_arxiv_found_refs[match_ref][0],
                'title': cache_arxiv_found_refs[match_ref][1],
                'raw_reference': match_ref
            })
    chunk_json['chunk_id'] = extract_arxiv_id(pdf_path)
    chunk_json['text'] = section['text']
    chunk_json['start_offset'] = None
    chunk_json['end_offset'] = None
    chunk_json['section'] = section['section'].lower()
    chunk_json['token_len'] = num_tokens_from_string(section['text'], "cl100k_base")
    chunk_json['embedding_id'] = 9999999
    if citations:
        chunk_json['is_citation_context'] = True
        chunk_json['citations'] = citations
    chunk_json['checksum'] = generate_checksum(section['text'])


    print(chunk_json)
    break