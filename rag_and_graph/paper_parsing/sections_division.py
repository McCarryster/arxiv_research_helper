from typing import List, Dict, Tuple, Optional
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import tiktoken
import html
import re


def _safe_text(node: ET.Element) -> str:
    """Extract text content from an XML element (including tail text), stripping extra whitespace."""
    texts = []
    if node.text:
        texts.append(node.text)
    for child in node:
        texts.append(_safe_text(child))
        if child.tail:
            texts.append(child.tail)
    return "".join(texts).strip()


def _extract_sections_from_tei(tei_xml: str) -> List[Tuple[str, str]]:
    """
    Parse TEI XML returned by GROBID and extract sections as (title, text).
    Strategy:
      - Find text/body and then <div> elements (sections). For each div, use its <head> as title
        (if present) and join all paragraph (<p>) text for section text.
      - Also handle divs without explicit head (use empty string or fallback).
    Returns list of (section_title, section_text) in document order.
    """
    # Parse with ElementTree
    # TEI uses namespaces; register common TEI namespace if present
    it = ET.ElementTree(ET.fromstring(tei_xml))
    root = it.getroot()

    # find namespace (if any)
    m = re.match(r"\{(.*)\}", root.tag) # type: ignore
    ns = {"tei": m.group(1)} if m else {}

    # locate body element
    body = None
    if ns:
        body = root.find(".//tei:text/tei:body", ns) # type: ignore
    else:
        body = root.find(".//text/body") # type: ignore

    if body is None:
        # fallback: consider any <body> tag
        body = root.find(".//body") # type: ignore

    sections: List[Tuple[str, str]] = []

    # find all divs under body (recursive)
    if body is None:
        return sections

    # collect div elements in order (depth-first)
    if ns:
        divs = body.findall(".//tei:div", ns)
    else:
        divs = body.findall(".//div")

    # If no divs found, try to gather paragraphs as a single section
    if not divs:
        paras = []
        if ns:
            paras = body.findall(".//tei:p", ns)
        else:
            paras = body.findall(".//p")
        combined = "\n\n".join([_safe_text(p) for p in paras if _safe_text(p)])
        if combined:
            sections.append(("Document", combined))
        return sections

    for div in divs:
        # find head/title
        head = None
        # title/head in TEI is usually <head>
        if ns:
            head = div.find("tei:head", ns)
        else:
            head = div.find("head")
        title = _safe_text(head) if head is not None else ""

        # gather paragraphs inside div
        if ns:
            paras = div.findall(".//tei:p", ns)
        else:
            paras = div.findall(".//p")
        # fallback: gather all text content inside div
        if not paras:
            text = _safe_text(div)
        else:
            text_parts = []
            for p in paras:
                p_text = _safe_text(p)
                if p_text:
                    text_parts.append(p_text)
            text = "\n\n".join(text_parts)

        # Clean HTML entities and excessive whitespace
        text = html.unescape(text).strip()
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if text:
            sections.append((title.strip() if title else "", text))

    return sections


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def chunk_pdf_with_grobid(
    pdf_path: str,
    grobid_url: str = "http://localhost:8070",
    token_threshold: int = 200,
    encoding_name: str = "cl100k_base",
    grobid_timeout: int = 120
) -> List[Dict[str, str]]:
    """
    Send PDF to GROBID, parse TEI into semantic sections, merge adjacent sections until
    token length >= token_threshold, and return list of dicts with keys:
      - section_title: str
      - section_text: str

    Parameters:
      pdf_path: local path to the PDF file to send to GROBID.
      grobid_url: base URL where GROBID is running (default localhost:8070).
      token_threshold: minimum token length for each returned chunk. Adjacent sections
                       will be merged until threshold is met. If threshold <= 0, no merging.
      encoding_name: encoding name passed to tiktoken (e.g., 'cl100k_base').
      grobid_timeout: seconds to wait for GROBID response.

    Returns:
      List of dictionaries: [{"section_title": "...", "section_text": "..."}, ...]
    """
    api_endpoint = f"{grobid_url.rstrip('/')}/api/processFulltextDocument"

    # Send PDF to GROBID
    with open(pdf_path, "rb") as f:
        files = {"input": (pdf_path, f, "application/pdf")}
        # options: consolidate headers/sections etc; keep default unless you want to change
        try:
            resp = requests.post(api_endpoint, files=files, timeout=grobid_timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Error {pdf_path} contacting GROBID at {api_endpoint}: {e}")

    tei_xml = resp.text

    # Parse TEI and extract raw sections
    raw_sections = _extract_sections_from_tei(tei_xml)

    # Merge small sections: iterate in order and append to buffer until buffer tokens >= threshold
    merged: List[Dict[str, str]] = []
    if token_threshold is None or token_threshold <= 0:
        # no merging; just compute and return
        for title, text in tqdm(raw_sections, desc="Preparing sections", unit="sec"):
            merged.append({"section_title": title, "section_text": text})
        return merged

    buffer_title = ""
    buffer_text = ""
    buffer_tokens = 0

    # for idx, (title, text) in enumerate(tqdm(raw_sections, desc="Merging sections", unit="sec")):
    for idx, (title, text) in enumerate(raw_sections):
        # If buffer empty, initialize with current section
        if not buffer_text:
            buffer_title = title or ""
            buffer_text = text or ""
            buffer_tokens = num_tokens_from_string(buffer_text, encoding_name)
            # If large enough already, push immediately
            if buffer_tokens >= token_threshold:
                merged.append({"section_title": buffer_title, "section_text": buffer_text})
                buffer_title = ""
                buffer_text = ""
                buffer_tokens = 0
            continue

        # Buffer exists but below threshold -> merge current section into it
        if buffer_tokens < token_threshold:
            # How to merge titles: prefer buffer_title if meaningful, otherwise use current title.
            # If both have titles, join them with " — " to preserve context.
            if buffer_title and title:
                new_title = f"{buffer_title} — {title}"
            elif buffer_title:
                new_title = buffer_title
            else:
                new_title = title or ""
            # join texts with double newline to keep paragraph boundaries
            new_text = buffer_text + "\n\n" + (text or "")
            # update buffer
            buffer_title = new_title
            buffer_text = new_text
            buffer_tokens = num_tokens_from_string(buffer_text, encoding_name)

            # if now meets threshold, flush
            if buffer_tokens >= token_threshold:
                merged.append({"section_title": buffer_title, "section_text": buffer_text})
                buffer_title = ""
                buffer_text = ""
                buffer_tokens = 0
            # else continue merging next sections
            continue

        # If buffer is already >= threshold (shouldn't happen because we flush immediately),
        # flush then start a new buffer.
        if buffer_tokens >= token_threshold:
            merged.append({"section_title": buffer_title, "section_text": buffer_text})
            buffer_title = title or ""
            buffer_text = text or ""
            buffer_tokens = num_tokens_from_string(buffer_text, encoding_name)
            if buffer_tokens >= token_threshold:
                merged.append({"section_title": buffer_title, "section_text": buffer_text})
                buffer_title = ""
                buffer_text = ""
                buffer_tokens = 0

    # End of loop: if buffer still contains text, flush it
    if buffer_text:
        merged.append({"section_title": buffer_title, "section_text": buffer_text})

    return merged


# pdf_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1308.0850v5.pdf"
# chunks = chunk_pdf_with_grobid(pdf_path, grobid_url="http://localhost:8070", token_threshold=256)
# encoding_name = "cl100k_base"
# for c in chunks:
#     print(c["section_title"])
#     print(c["section_text"])
#     print(num_tokens_from_string(c["section_text"], encoding_name))
#     print("#"*120)