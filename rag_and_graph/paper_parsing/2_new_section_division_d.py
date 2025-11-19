# Function should return
#     section_title: str
#     section_text: str
#     start_offset: int
#     end_offset: int
#     start_offset and end_offset specify the exact position of the chunk’s text within the full pdf paper, enabling precise reference and retrieval. So using those vales I should be able to find that exact chunk within pdf paper


from typing import List, Dict, Tuple, Optional
import re
import requests
from lxml import etree # type: ignore
from tqdm import tqdm
import fitz  # PyMuPDF
from difflib import SequenceMatcher
from pypdf import PdfReader

# ---- Helper types ----
Section = Dict[str, object]  # {section_title: str, section_text: str, start_offset: int, end_offset: int}


def pdf_to_sections_with_offsets(
    pdf_path: str,
    grobid_url: str = "http://localhost:8070",
    grobid_timeout: int = 120,
    min_chars: int = 200,
    fuzzy_match_threshold: float = 0.85,
) -> List[Section]:
    """
    Main function.

    Parameters
    ----------
    pdf_path:
        Path to the PDF arXiv paper file.
    grobid_url:
        Base URL for the GROBID service (default: http://localhost:8070).
        Assumes GROBID is running and accessible.
    grobid_timeout:
        Seconds to wait for a grobid response.
    min_chars:
        Minimum section length in characters. Sections shorter than this will be merged
        into neighboring sections (prefer merging forward, otherwise backward).
    fuzzy_match_threshold:
        If exact normalized string search fails, SequenceMatcher ratio above this
        on a sampled window will be considered a match.

    Returns
    -------
    List[dict] with keys:
        - section_title: str
        - section_text: str
        - start_offset: int  # start char index in the raw full-text extracted from PDF
        - end_offset: int    # end char index (exclusive)
    """
    # 1) call GROBID to get TEI XML
    tei = _call_grobid_process_fulltext(pdf_path, grobid_url=grobid_url, timeout=grobid_timeout)

    # 2) parse TEI into semantic sections (title + text)
    raw_sections = _parse_tei_sections(tei)

    # 3) extract full text from PDF (raw)
    full_text = _extract_text_from_pdf(pdf_path)

    # 4) create normalized version of full_text with mapping to original indexes
    full_norm, full_map = _normalize_with_mapping(full_text)

    # 5) for each section, normalize its text and locate within full_norm -> map back to original offsets
    sections_with_offsets: List[Section] = []
    for sec in tqdm(raw_sections, desc="Locating sections in full text"):
        title = sec["title"] or ""
        text = sec["text"] or ""
        sec_norm = _normalize_text_for_match(text)

        start_off, end_off = _find_normalized_span_in_full(
            sec_norm, full_norm, full_map, full_text, fuzzy_threshold=fuzzy_match_threshold
        )

        sections_with_offsets.append(
            {
                "section_title": title,
                "section_text": text,
                "start_offset": start_off,
                "end_offset": end_off,
            }
        )

    # 6) Merge small sections (by character count) where needed
    merged = _merge_small_sections(sections_with_offsets, min_chars=min_chars)

    return merged


# --------------------- GROBID interaction ---------------------
def _call_grobid_process_fulltext(pdf_path: str, grobid_url: str, timeout: int) -> str:
    """
    POST the PDF to GROBID's /api/processFulltextDocument and return TEI XML as string.
    Requires GROBID running locally (or at given URL).
    """
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        files = {"input": (pdf_path, f, "application/pdf")}
        # request parameters: we keep defaults; adjust if your GROBID needs specific params
        resp = requests.post(endpoint, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.text


# --------------------- TEI parsing ---------------------
def _parse_tei_sections(tei_xml: str) -> List[Dict[str, str]]:
    """
    Parse TEI XML returned by GROBID and return ordered list of sections:
    [{ "title": str, "text": str }, ...]
    This function is tolerant of namespaces and nested divs.
    """
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(tei_xml.encode("utf-8"), parser=parser)

    # Find the <text> element if present, then <body>, then collect <div> children that are sections
    # We'll walk through body and collect each <div> that looks like a logical section.
    # Namespace-agnostic search:
    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1]

    text_elem = None
    for el in root.iter():
        if strip_ns(el.tag).lower() == "text":
            text_elem = el
            break

    if text_elem is None:
        # no TEI text element; fallback: use entire document text content
        doc_text = "".join(root.itertext()).strip()
        return [{"title": "document", "text": doc_text}]

    body_elem = None
    for el in text_elem.iterchildren():
        if strip_ns(el.tag).lower() == "body":
            body_elem = el
            break

    if body_elem is None:
        # fallback to text element content
        doc_text = "".join(text_elem.itertext()).strip()
        return [{"title": "document", "text": doc_text}]

    sections: List[Dict[str, str]] = []

    # We will treat each top-level div under body as a section (preserves order).
    for div in body_elem:
        if strip_ns(div.tag).lower() != "div":
            # maybe stray text or p: if it contains p text, include as a fallback section
            if "".join(div.itertext()).strip():
                sections.append({"title": "", "text": "".join(div.itertext()).strip()})
            continue

        # title/head
        title = ""
        # body text: collect all paragraph (<p>) text, or fallback to div.itertext()
        p_texts = []
        for child in div.iter():
            tname = strip_ns(child.tag).lower()
            if tname == "head":
                if child.text and child.text.strip():
                    title = (title + " " + child.text.strip()).strip()
            elif tname == "p":
                chunk = "".join(child.itertext()).strip()
                if chunk:
                    p_texts.append(chunk)
        if not p_texts:
            # fallback: take the div text excluding nested head
            div_text = []
            for node in div.iter():
                if strip_ns(node.tag).lower() == "head":
                    continue
                if node.text:
                    div_text.append(node.text)
            joined = " ".join(t.strip() for t in div_text if t and t.strip())
            joined = joined.strip()
            if joined:
                p_texts = [joined]

        section_text = "\n\n".join(p_texts).strip()
        sections.append({"title": title, "text": section_text})

        # Also consider grandchildren divs (subsections) by iterating children that are divs
        for child in div:
            if strip_ns(child.tag).lower() == "div":
                title_sub = ""
                p_texts_sub = []
                for node in child.iter():
                    if strip_ns(node.tag).lower() == "head":
                        if node.text and node.text.strip():
                            title_sub = (title_sub + " " + node.text.strip()).strip()
                    elif strip_ns(node.tag).lower() == "p":
                        chunk = "".join(node.itertext()).strip()
                        if chunk:
                            p_texts_sub.append(chunk)
                if p_texts_sub:
                    sections.append({"title": title_sub or "", "text": "\n\n".join(p_texts_sub).strip()})

    # Filter out any empty sections
    filtered = [s for s in sections if s["text"].strip()]
    return filtered


# --------------------- PDF text extraction ---------------------
def _extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from PDF using PyMuPDF (fitz). Collapses pages into a single string
    separated by newline markers between pages.
    """
    doc = fitz.open(pdf_path)
    page_texts: List[str] = []
    for page in tqdm(doc, desc="Extracting text from PDF (pages)"): # type: ignore
        # get_text("text") returns plain textual representation
        text = page.get_text("text")
        page_texts.append(text)
    doc.close()
    # Join pages with a page break marker (single newline is ok for offsets mapping)
    return "\n".join(page_texts)


# --------------------- Normalization & mapping ---------------------
def _normalize_text_for_match(s: str) -> str:
    """
    Normalize a section text for matching:
      - remove hyphenation at line breaks (e.g., 'exam-\nple' -> 'example')
      - replace newlines with spaces
      - collapse whitespace to single spaces
      - strip leading/trailing spaces
    """
    # Remove hyphenation at line breaks: '-\n' or '-\r\n' possibly with spaces
    s = re.sub(r"-\s*\n\s*", "", s)
    # Replace remaining newlines with spaces
    s = re.sub(r"\s*\n\s*", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalize_with_mapping(original: str) -> Tuple[str, List[int]]:
    """
    Normalize the original full text in the same way as _normalize_text_for_match
    but build and return a mapping from normalized index -> original index.

    Returns:
      normalized_string, mapping_list
      mapping_list[i] == index in original corresponding to normalized_string[i]
    """
    # 1) remove hyphenation at line breaks
    # We will work from the original, applying the same transforms but tracking indices.
    orig = original

    # Replace hyphenation at line breaks by removing the hyphen and newline.
    # But to keep mapping accurate, we'll create a new string `pre` and a mapping from pre-index -> orig-index.
    pre = []
    pre_map = []  # pre_map[i] = index in original for pre[i]
    i = 0
    L = len(orig)
    while i < L:
        # Detect pattern: '-' followed by optional spaces then newline
        if orig[i] == "-" and i + 1 < L:
            # look ahead to see if there is a hyphenation pattern
            j = i + 1
            # skip spaces
            while j < L and orig[j].isspace() and orig[j] not in ("\n", "\r"):
                j += 1
            # if there's a newline following (or immediately whitespace+newline), treat as hyphenation
            if j < L and orig[j] in ("\n", "\r"):
                # skip the '-' and the immediate newline/CR and continue (i jumps to j+1)
                # we simply drop the '-' and the newline(s) to join hyphenated words
                # map nothing for the dropped chars (they are omitted)
                # advance i to the character after newline(s)
                # move j to after any combination of \r\n
                k = j
                # skip the newline and any additional newline characters
                while k < L and orig[k] in ("\n", "\r"):
                    k += 1
                i = k
                continue
        # normal char: append and map
        pre.append(orig[i])
        pre_map.append(i)
        i += 1

    pre_str = "".join(pre)

    # Next: replace newline sequences with single space, but maintain mapping.
    norm_chars: List[str] = []
    norm_map: List[int] = []
    i = 0
    L = len(pre_str)
    while i < L:
        ch = pre_str[i]
        if ch in ("\n", "\r") or (ch.isspace() and ch != " "):
            # collapse sequence of whitespace into single space
            # find first index in original (pre_map) that corresponds to the sequence start
            start_idx = pre_map[i]
            # advance through sequence
            j = i
            while j < L and pre_str[j].isspace():
                j += 1
            # append one space and give it mapping to the start char index
            norm_chars.append(" ")
            norm_map.append(start_idx)
            i = j
        else:
            norm_chars.append(ch)
            norm_map.append(pre_map[i])
            i += 1

    # Collapse multiple spaces into single space (we already did for newlines but there may be sequences)
    final_chars: List[str] = []
    final_map: List[int] = []
    i = 0
    L = len(norm_chars)
    while i < L:
        ch = norm_chars[i]
        if ch == " ":
            # append one space and advance through any subsequent spaces
            final_chars.append(" ")
            final_map.append(norm_map[i])
            j = i + 1
            while j < L and norm_chars[j] == " ":
                j += 1
            i = j
        else:
            final_chars.append(ch)
            final_map.append(norm_map[i])
            i += 1

    final_str = "".join(final_chars).strip()
    # If we stripped leading/trailing spaces, adjust mapping: find first non-space index
    # we have to ensure the mapping aligns with final_str indices. The strip removes possible first/last space.
    # If final_str is empty, return empty mapping too.
    if not final_str:
        return "", []

    # Build mapping trimmed to the stripped string
    # Find first non-space in final_chars
    first_non_space = 0
    while first_non_space < len(final_chars) and final_chars[first_non_space] == " ":
        first_non_space += 1
    last_non_space = len(final_chars) - 1
    while last_non_space >= 0 and final_chars[last_non_space] == " ":
        last_non_space -= 1

    trimmed_map = final_map[first_non_space : last_non_space + 1]
    trimmed_str = "".join(final_chars[first_non_space : last_non_space + 1])

    return trimmed_str, trimmed_map


# --------------------- Matching normalized section -> full normalized ---------------------
def _find_normalized_span_in_full(
    sec_norm: str,
    full_norm: str,
    full_map: List[int],
    full_original: str,
    fuzzy_threshold: float = 0.85,
) -> Tuple[int, int]:
    """
    Find the (start, end) offsets in the original full text for a normalized section string.
    Returns (-1, -1) if no plausible mapping could be found.

    Strategy:
      - Try direct substring search on normalized forms.
      - If not found, try searching for the first 120 characters (or full length if smaller).
      - If still not found, perform a sampled sliding-window similarity search using SequenceMatcher
        and accept if ratio >= fuzzy_threshold.
    """
    if not sec_norm.strip():
        return -1, -1

    # 1) direct find
    idx = full_norm.find(sec_norm)
    if idx != -1:
        start_original = full_map[idx]
        end_norm_idx = idx + len(sec_norm) - 1
        end_original = full_map[end_norm_idx] + 1  # exclusive
        return start_original, end_original

    # 2) try searching for a short snippet from the beginning of sec_norm (handle minor truncation)
    snippet_len = min(160, len(sec_norm))
    snippet = sec_norm[:snippet_len]
    idx2 = full_norm.find(snippet)
    if idx2 != -1:
        # attempt to expand match forward and backward to better approximate boundaries
        start_norm = idx2
        end_norm = idx2 + len(sec_norm)
        # clamp end_norm
        if end_norm > len(full_norm):
            end_norm = len(full_norm)
        # map to original offsets
        start_original = full_map[start_norm]
        end_original = full_map[end_norm - 1] + 1
        return start_original, end_original

    # 3) sampled sliding window fuzzy matching
    # We'll compare the first up-to-500 chars of sec_norm to sliding windows of the same length in full_norm,
    # sampled every `step`. Keep the best ratio.
    sample = sec_norm[:500] if len(sec_norm) > 500 else sec_norm
    window_len = len(sample)
    if window_len == 0 or len(full_norm) < 10:
        return -1, -1

    best_ratio = 0.0
    best_idx = -1
    # step chosen so this loop is practical for large texts; tqdm not used here to avoid too much output
    step = max(1, min(200, window_len // 3))
    for start in range(0, max(1, len(full_norm) - window_len + 1), step):
        window = full_norm[start : start + window_len]
        ratio = SequenceMatcher(None, sample, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = start
        # early accept
        if best_ratio >= fuzzy_threshold:
            break

    if best_ratio >= fuzzy_threshold and best_idx != -1:
        # map approximate start and end; map entire section by assuming placement at best_idx
        start_norm = best_idx
        # attempt to determine end_norm by searching for the last few chars of the section near the guessed area
        guessed_end_norm = min(len(full_norm), start_norm + len(sec_norm))
        start_original = full_map[start_norm]
        end_original = full_map[guessed_end_norm - 1] + 1
        return start_original, end_original

    # 4) Give up with -1
    return -1, -1


# --------------------- Merge small sections ---------------------
def _merge_small_sections(sections: List[Section], min_chars: int = 200) -> List[Section]:
    """
    Merge sections whose text length is less than min_chars into their neighbors.
    Strategy:
      - Iterate left-to-right. If a section is too small, merge with the next section (if exists).
      - If it's the last section and too small, merge with previous.
      - Preserve chronological order. Recompute start/end as min(start) and max(end).
    """
    if not sections:
        return []

    merged: List[Section] = []
    i = 0
    while i < len(sections):
        sec = sections[i]
        text_len = len(sec["section_text"] or "") # type: ignore
        if text_len >= min_chars:
            # keep as is
            merged.append(sec.copy())
            i += 1
            continue

        # too small: try merge with next if possible
        if i + 1 < len(sections):
            nxt = sections[i + 1]
            combined_text = (sec["section_text"] or "") + "\n\n" + (nxt["section_text"] or "") # type: ignore
            combined_title = sec["section_title"] or nxt["section_title"]
            start_candidates = [x for x in (sec["start_offset"], nxt["start_offset"]) if isinstance(x, int) and x >= 0]
            end_candidates = [x for x in (sec["end_offset"], nxt["end_offset"]) if isinstance(x, int) and x >= 0]
            start_off = min(start_candidates) if start_candidates else -1
            end_off = max(end_candidates) if end_candidates else -1
            merged.append(
                {
                    "section_title": combined_title,
                    "section_text": combined_text,
                    "start_offset": start_off,
                    "end_offset": end_off,
                }
            )
            # skip the next one as it's merged
            i += 2
        else:
            # last item and too small: merge with previous (if any)
            if merged:
                prev = merged.pop()
                combined_text = (prev["section_text"] or "") + "\n\n" + (sec["section_text"] or "") # type: ignore
                combined_title = prev["section_title"] or sec["section_title"]
                start_candidates = [x for x in (prev["start_offset"], sec["start_offset"]) if isinstance(x, int) and x >= 0]
                end_candidates = [x for x in (prev["end_offset"], sec["end_offset"]) if isinstance(x, int) and x >= 0]
                start_off = min(start_candidates) if start_candidates else -1
                end_off = max(end_candidates) if end_candidates else -1
                merged.append(
                    {
                        "section_title": combined_title,
                        "section_text": combined_text,
                        "start_offset": start_off,
                        "end_offset": end_off,
                    }
                )
            else:
                # nothing to merge with -- keep as-is
                merged.append(sec.copy())
            i += 1

    # Final sanity: ensure offsets are ints and start <= end or -1
    for s in merged:
        s["start_offset"] = int(s["start_offset"]) if isinstance(s["start_offset"], int) else -1
        s["end_offset"] = int(s["end_offset"]) if isinstance(s["end_offset"], int) else -1
        if s["start_offset"] >= 0 and s["end_offset"] >= 0 and s["start_offset"] > s["end_offset"]:
            # swap if inverted
            s["start_offset"], s["end_offset"] = s["end_offset"], s["start_offset"]
    return merged


pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf"
chunks = pdf_to_sections_with_offsets(pdf_path, grobid_url="http://localhost:8070", min_chars=500)
for ch in chunks:
    # print(ch["section_title"], ch["start_offset"], ch["end_offset"])
    print(ch)
    print('#'*120)


print()
print()
print()
# 2874, 'end_offset': 4777
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
text_chunk = extract_text_by_offset(pdf_path, 2874, 4777)
print(text_chunk)
