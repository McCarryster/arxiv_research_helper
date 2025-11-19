# Function should return
#     section_title: str
#     section_text: str
#     start_offset: int
#     end_offset: int
#     start_offset and end_offset specify the exact position of the chunk’s text within the full pdf paper, enabling precise reference and retrieval. So using those vales I should be able to find that exact chunk within pdf paper


"""
Split an arXiv PDF into semantically meaningful chunks using GROBID, merge small chunks,
and return chunk text with start/end offsets into the full document text.

Dependencies:
- requests
- lxml
- tqdm

Install if needed:
pip install requests lxml tqdm
"""

from typing import List, Dict, Optional, Tuple
import requests
from lxml import etree # type: ignore
import io
import re
from difflib import SequenceMatcher
from tqdm import tqdm
from pypdf import PdfReader


def _collapse_whitespace(s: str) -> str:
    """Collapse all whitespace sequences into single spaces and strip ends."""
    return re.sub(r'\s+', ' ', s).strip()


def _build_norm_mappings(orig: str) -> Tuple[str, List[int]]:
    """
    Build normalized string (collapsed whitespace) and mapping from normalized char index
    -> original char index (the position in orig of the corresponding normalized char).
    Returns (normalized_string, norm_to_orig_index_list).
    """
    norm_chars = []
    norm_to_orig = []
    i = 0
    length = len(orig)
    while i < length:
        ch = orig[i]
        if ch.isspace():
            # collapse run of whitespace to a single space
            # but only add a single space if last added char isn't a space
            while i < length and orig[i].isspace():
                i += 1
            if len(norm_chars) == 0 or norm_chars[-1] != ' ':
                norm_chars.append(' ')
                # map this normalized char to the original index where the space run started
                # choose the index of the first whitespace in the run (approximate)
                # we could also map to the last index of the run - either is acceptable
                norm_to_orig.append(i - 1)
        else:
            norm_chars.append(ch)
            norm_to_orig.append(i)
            i += 1
    norm_str = ''.join(norm_chars).strip()
    # If we stripped leading/trailing spaces, the mapping must reflect the first/last non-space
    # Find first non-space original idx for start offset correction
    return norm_str, norm_to_orig


def _find_offsets_for_chunk(full_text: str, chunk_text: str) -> Tuple[int, int, str]:
    """
    Try to find exact start/end offsets of chunk_text inside full_text.
    Returns (start_offset, end_offset, method) where method indicates how found:
      - "exact": direct substring find
      - "normalized": found in whitespace-normalized text and mapped back
      - "fuzzy": used SequenceMatcher to find best matching block (best-effort)
    If not found at all, raises ValueError.
    """
    # 1) Try exact find
    start = full_text.find(chunk_text)
    if start != -1:
        return start, start + len(chunk_text), "exact"

    # 2) Try whitespace-collapsed normalized find and map indices
    norm_full, norm_to_orig = _build_norm_mappings(full_text)
    norm_chunk = _collapse_whitespace(chunk_text)

    if len(norm_chunk) == 0:
        raise ValueError("Chunk text is empty after normalization.")

    idx = norm_full.find(norm_chunk)
    if idx != -1:
        # map normalized index to original index
        # start original index is norm_to_orig[idx]
        start_orig = norm_to_orig[idx]
        # end original index: map the last char of the normalized match back
        # normalized match covers len(norm_chunk) characters; last char index is idx + len(norm_chunk) - 1
        last_norm_idx = idx + len(norm_chunk) - 1
        end_orig = norm_to_orig[last_norm_idx] + 1  # +1 because end offset is exclusive
        # As small adjustment, we can expand to include leading/trailing whitespace that was stripped
        return start_orig, end_orig, "normalized"

    # 3) Fallback: SequenceMatcher best block (best-effort)
    sm = SequenceMatcher(None, full_text, chunk_text)
    # get_matching_blocks yields blocks; choose the largest contiguous block
    blocks = sm.get_matching_blocks()
    best_block = max(blocks, key=lambda b: b.size) if blocks else None
    if best_block and best_block.size > 30:
        # best matched block: a = index in full_text, size = match length
        start_best = best_block.a
        end_best = start_best + best_block.size
        # Try to expand a bit around the matched block to capture full chunk if contiguous
        # We'll extend up to boundaries but not beyond chunk_text length
        # This is an approximation; caller should be aware it is best-effort.
        return start_best, end_best, "fuzzy"

    # nothing found
    raise ValueError("Could not locate chunk inside full_text (exact/norm/fuzzy failed).")


def split_pdf_into_sections_with_grobid(
    pdf_path: str,
    grobid_url: str = "http://localhost:8070",
    min_chars: int = 300,
    merge_forward: bool = True,
    consolidate_header: bool = True,
    timeout: int = 120
) -> List[Dict[str, object]]:
    """
    Process a PDF using GROBID, split into semantic sections, merge small sections,
    and return chunks with exact start/end offsets into the full document text.

    Args:
        pdf_path: path to local PDF file (arXiv paper).
        grobid_url: base URL of your GROBID server (default: http://localhost:8070).
        min_chars: minimum number of characters a chunk should have. If a chunk is smaller,
                   it will be merged (forward by default) until the threshold is reached.
        merge_forward: if True merge small chunk into following chunk(s). If False, merge backwards.
        consolidate_header: pass to GROBID to try to consolidate header information (True by default).
        timeout: HTTP timeout in seconds for the GROBID request.

    Returns:
        List of dicts: each dict has keys:
            - section_title: str
            - section_text: str
            - start_offset: int (inclusive)
            - end_offset: int (exclusive)

    Notes:
        - This function uses the text extracted by GROBID (TEI). Offsets refer to positions in
          the concatenated TEI body text produced here.
        - If exact matching fails for a chunk, the function attempts normalized matching
          (whitespace collapsed), then a best-effort fuzzy match.
    """
    # 1) Send PDF to GROBID
    process_endpoint = f"{grobid_url.rstrip('/')}/api/processFulltextDocument"
    files = {"input": open(pdf_path, "rb")}
    data = {}
    if consolidate_header:
        data["consolidateHeader"] = "1"

    resp = requests.post(process_endpoint, files=files, data=data, timeout=timeout)
    files["input"].close()
    resp.raise_for_status()
    tei_xml = resp.content

    # 2) Parse TEI XML and extract body text and sections
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(tei_xml, parser=parser)

    # GROBID TEI typically uses the TEI namespace; handle with or without namespace
    nsmap = root.nsmap
    # Find namespace for TEI if present
    tei_ns = None
    for k, v in nsmap.items():
        if v and "tei" in (k or "") or (v and 'http://www.tei-c.org' in v):
            tei_ns = v
            break
    # Build namespace mapping for XPath
    ns = {'tei': tei_ns} if tei_ns else {}

    # Helper to build xpath strings that work both with and without namespace
    def xp(path: str) -> str:
        if tei_ns:
            # replace element names with tei:element
            return '/'.join('tei:' + p for p in path.split('/'))
        else:
            return path

    # Extract full body text (concatenate text from <body>)
    body_nodes = root.xpath(xp("TEI/text/body")) if root.xpath(xp("TEI/text/body"), namespaces=ns) else root.xpath(xp("text/body"), namespaces=ns)
    if not body_nodes:
        # try alternate path
        body_nodes = root.xpath(".//tei:body", namespaces=ns) if tei_ns else root.xpath(".//body")
    if not body_nodes:
        raise RuntimeError("Could not find <body> in GROBID TEI output.")

    body_node = body_nodes[0]
    full_text = ''.join(body_node.itertext())
    # Normalize line endings but preserve original characters for offsets
    # We will not further collapse whitespace for full_text now; offsets refer to this `full_text`
    full_text = full_text.replace('\r\n', '\n').replace('\r', '\n')

    # Extract semantic sections: find all div elements with type='section' under body (maintain order)
    # Use XPath to get all divs under body (recursive), type='section' if available
    if tei_ns:
        divs = body_node.xpath(".//tei:div", namespaces=ns)
    else:
        divs = body_node.xpath(".//div")

    sections: List[Dict[str, str]] = []
    for d in divs:
        # Some divs are not 'section' type; keep any div that contains meaningful text content
        # Get head text as title, paragraphs as content
        head = None
        if tei_ns:
            head_nodes = d.xpath("./tei:head", namespaces=ns)
        else:
            head_nodes = d.xpath("./head")
        if head_nodes:
            head = ' '.join([_collapse_whitespace(''.join(h.itertext())) for h in head_nodes]).strip()
            if head == '':
                head = None

        # Collect textual content from paragraph elements inside this div (<p>, <list>, etc.)
        txt_parts = []
        if tei_ns:
            # collect paragraphs, p, list items, note that sometimes text sits directly within div
            p_nodes = d.xpath(".//tei:p", namespaces=ns)
        else:
            p_nodes = d.xpath(".//p")
        if p_nodes:
            for p in p_nodes:
                t = ''.join(p.itertext()).strip()
                if t:
                    txt_parts.append(_collapse_whitespace(t))
        else:
            # fallback: use all text in div
            t = ''.join(d.itertext()).strip()
            if t:
                txt_parts.append(_collapse_whitespace(t))

        section_text = ' '.join(txt_parts).strip()
        if section_text:
            sections.append({"title": head or "", "text": section_text})

    # If no div sections were found, fallback to splitting body into paragraphs
    if not sections:
        if tei_ns:
            p_nodes = body_node.xpath(".//tei:p", namespaces=ns)
        else:
            p_nodes = body_node.xpath(".//p")
        for p in p_nodes:
            t = ''.join(p.itertext()).strip()
            if t:
                sections.append({"title": "", "text": _collapse_whitespace(t)})

    # 3) Merge small sections according to min_chars
    merged_sections: List[Dict[str, str]] = []
    i = 0
    N = len(sections)
    # Use tqdm for iteration per your instruction (applies for long lists)
    iter_range = range(N)
    for idx in tqdm(iter_range, desc="Merging small sections", leave=False):
        if idx < i:
            continue
        cur_title = sections[idx]["title"]
        cur_text = sections[idx]["text"]
        # If chunk already large enough, keep
        if len(cur_text) >= min_chars:
            merged_sections.append({"title": cur_title, "text": cur_text})
            i = idx + 1
            continue

        # Need to merge
        if merge_forward:
            # merge forward into subsequent sections until threshold reached
            j = idx + 1
            merged_text = cur_text
            merged_title = cur_title or ""
            while j < N and len(merged_text) < min_chars:
                # append a separator (space) to keep words intact
                next_title = sections[j]["title"]
                next_text = sections[j]["text"]
                # Update title if current has no title and next has one, or combine them
                if not merged_title and next_title:
                    merged_title = next_title
                elif next_title:
                    merged_title = merged_title + " | " + next_title if merged_title else next_title
                merged_text = merged_text + " " + next_text
                j += 1
            merged_sections.append({"title": merged_title, "text": _collapse_whitespace(merged_text)})
            i = j
        else:
            # merge backward (merge into previous chunk if exists), otherwise forward
            if merged_sections:
                # pop last and merge
                prev = merged_sections.pop()
                prev_title = prev["title"]
                prev_text = prev["text"]
                new_title = prev_title or cur_title
                new_text = prev_text + " " + cur_text
                # if still short, try to also consume following until threshold reached
                j = idx + 1
                while len(new_text) < min_chars and j < N:
                    next_title = sections[j]["title"]
                    next_text = sections[j]["text"]
                    if next_title:
                        new_title = new_title + " | " + next_title if new_title else next_title
                    new_text = new_text + " " + next_text
                    j += 1
                merged_sections.append({"title": new_title, "text": _collapse_whitespace(new_text)})
                i = j
            else:
                # no previous to merge into; fallback to forward merging
                j = idx + 1
                merged_text = cur_text
                merged_title = cur_title or ""
                while j < N and len(merged_text) < min_chars:
                    next_title = sections[j]["title"]
                    next_text = sections[j]["text"]
                    if not merged_title and next_title:
                        merged_title = next_title
                    elif next_title:
                        merged_title = merged_title + " | " + next_title if merged_title else next_title
                    merged_text = merged_text + " " + next_text
                    j += 1
                merged_sections.append({"title": merged_title, "text": _collapse_whitespace(merged_text)})
                i = j

    # 4) For each merged section, locate offsets in the full_text
    result_chunks: List[Dict[str, object]] = []
    for sec in tqdm(merged_sections, desc="Locating chunk offsets", leave=False):
        title = sec["title"]
        text = sec["text"]
        try:
            start, end, method = _find_offsets_for_chunk(full_text, text)
        except ValueError:
            # extremely unlikely, but as fallback put chunk at end with approximate offsets
            # produce best-effort: try to find a short prefix
            prefix = text[:200]
            pos = full_text.find(prefix)
            if pos != -1:
                start = pos
                # attempt to find end using next words
                end = min(len(full_text), start + len(text))
                method = "approx_prefix"
            else:
                # last resort: append at end
                start = len(full_text)
                end = start + len(text)
                method = "appended_at_end"
        result_chunks.append({
            "section_title": title,
            "section_text": text,
            "start_offset": int(start),
            "end_offset": int(end),
            "match_method": method  # extra field for debugging; remove if you want strict output
        })

    # Optionally remove 'match_method' field before returning if strict schema is required
    # The user requested the function should return section_title, section_text, start_offset, end_offset.
    # We'll return only those four fields for each chunk.
    final = []
    for c in result_chunks:
        final.append({
            "section_title": c["section_title"],
            "section_text": c["section_text"],
            "start_offset": c["start_offset"],
            "end_offset": c["end_offset"]
        })

    return final


pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf"
chunks = split_pdf_into_sections_with_grobid(pdf_path, grobid_url="http://localhost:8070", min_chars=500)
for ch in chunks:
    # print(ch["section_title"], ch["start_offset"], ch["end_offset"])
    print(ch)
    print('#'*120)


print()
print()
print()
def extract_text_by_offset(pdf_path, start_offset, end_offset):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    # Safely slice text within bounds
    start = max(0, start_offset)
    end = min(len(full_text), end_offset)
    return full_text[start:end]
pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf"
text_chunk = extract_text_by_offset(pdf_path, 10850, 11351)
print(text_chunk)
