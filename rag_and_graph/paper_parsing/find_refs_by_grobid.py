import re
import io
import os
from typing import List, Dict, Optional, Union
import requests
import xml.etree.ElementTree as ET

# ----------------------------
# Utility helpers
# ----------------------------
def _get_tei_ns(root: ET.Element) -> str:
    if root.tag.startswith("{"):
        return root.tag.split("}")[0].strip("{")
    return ""

def _text_of_element(elem: Optional[ET.Element]) -> str:
    return "".join(elem.itertext()).strip() if elem is not None else ""

def _safe_find(elem: ET.Element, xpath: str, ns: dict) -> Optional[ET.Element]:
    try:
        return elem.find(xpath, namespaces=ns)
    except Exception:
        return None

def _safe_findall(elem: ET.Element, xpath: str, ns: dict) -> List[ET.Element]:
    try:
        return elem.findall(xpath, namespaces=ns)
    except Exception:
        return []

def _clean_whitespace(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r'\s+', ' ', s).strip()

# ----------------------------
# GROBID HTTP helper
# ----------------------------
def _call_grobid(grobid_url: str, endpoint: str, pdf_bytes: bytes, data: dict, timeout: int = 120) -> bytes:
    """
    POST PDF bytes to a GROBID endpoint and return response content (XML/TEI).
    endpoint example: "/api/processFulltextDocument" or "/api/processReferences"
    """
    url = grobid_url.rstrip("/") + endpoint
    files = {"input": ("file.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    headers = {"Accept": "application/xml"}
    resp = requests.post(url, files=files, data=data, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.content

# ----------------------------
# Author / TEI heuristics
# ----------------------------
def _extract_authors_from_raw(raw: str) -> Optional[str]:
    """
    Try to extract the author block from the beginning of a reference by taking text up to
    the year or up to a parenthesized year. Returns a cleaned string or None.
    """
    if not raw:
        return None
    raw = raw.strip()
    m = re.match(r'^(?P<authors>.+?)(?:\.\s*(?:19|20)\d{2}|\s*\((?:19|20)\d{2}\))', raw)
    if m:
        return _clean_whitespace(m.group("authors"))
    first_sentence = raw.split('.', 1)[0]
    if 1 < len(first_sentence.split()) <= 8:
        return _clean_whitespace(first_sentence)
    return None

def _split_authors_string(authors_str: str) -> List[str]:
    """Conservative splitting of an authors string into names."""
    if not authors_str:
        return []
    s = authors_str.strip()
    parts = re.split(r'\s+and\s+|;\s*|(?<=\w),\s+(?=[A-Z][a-z])', s)
    return [p.strip().rstrip(',') for p in parts if p.strip()]

def _parse_persname_elements(item: ET.Element, ns: dict) -> List[Dict]:
    """
    Return list of dicts {forename, surname, raw} for each persName inside item.
    """
    pers = []
    for pn in _safe_findall(item, ".//tei:persName", ns):
        fn_elem = _safe_find(pn, "tei:forename", ns)
        sn_elem = _safe_find(pn, "tei:surname", ns)
        forename = _text_of_element(fn_elem)
        surname = _text_of_element(sn_elem)
        raw = _text_of_element(pn)
        pers.append({"forename": forename, "surname": surname, "raw": raw})
    return pers

def _heuristic_merge_persnames(pers_list: List[Dict], raw_authors_str: Optional[str]) -> List[str]:
    """
    Merge broken persName pieces with heuristics:
      e.g. [{'forename':'D','surname':'Matthew'}, {'forename':'','surname':'Zeiler'}]
    -> "Matthew D. Zeiler" if raw authors string suggests that order.
    """
    out = []
    i = 0
    L = len(pers_list)
    while i < L:
        cur = pers_list[i]
        # both present: straightforward
        if cur.get("forename") and cur.get("surname"):
            assembled = f"{cur['forename']} {cur['surname']}".strip()
            out.append(_clean_whitespace(assembled))
            i += 1
            continue

        # attempt merge with next if pattern matches
        if i + 1 < L:
            nxt = pers_list[i + 1]
            if (cur.get("forename") and len(cur["forename"].strip()) <= 2
                and cur.get("surname")
                and nxt.get("surname") and not nxt.get("forename")):
                # construct candidate "Surname ForenameInitial. NextSurname"
                initial = cur["forename"].strip()
                initial_display = initial + '.' if not initial.endswith('.') else initial
                cand = f"{cur['surname']} {initial_display} {nxt['surname']}"
                cand2 = f"{cur['surname']} {initial} {nxt['surname']}"
                raw_lower = (raw_authors_str or "").lower()
                if cand.lower() in raw_lower or cand2.lower() in raw_lower or not raw_authors_str:
                    out.append(_clean_whitespace(cand))
                    i += 2
                    continue
        # fallback cases
        if cur.get("surname") and not cur.get("forename"):
            out.append(_clean_whitespace(cur["surname"]))
            i += 1
            continue
        if cur.get("forename") and not cur.get("surname"):
            out.append(_clean_whitespace(cur["forename"]))
            i += 1
            continue
        if cur.get("raw"):
            out.append(_clean_whitespace(cur["raw"]))
        i += 1
    # final cleanup: remove empties
    return [x for x in out if x]

# ----------------------------
# TEI parsing
# ----------------------------
def parse_references_from_tei_bytes(tei_bytes: bytes) -> List[Dict]:
    """
    Given TEI bytes from GROBID, extract references with improved author handling.

    Returns list of dicts with keys:
      id, raw, xml, title, year, doi, venue, authors_teiled, authors_raw_str, authors_raw_list
    """
    try:
        root = ET.fromstring(tei_bytes)
    except ET.ParseError:
        return []

    tei_ns = _get_tei_ns(root)
    ns = {'tei': tei_ns} if tei_ns else {}

    # prioritize explicit references div
    listbibs = _safe_findall(root, ".//tei:div[@type='references']//tei:listBibl", ns)
    if not listbibs:
        listbibs = _safe_findall(root, ".//tei:listBibl", ns)

    results = []
    idx = 1
    for lb in listbibs:
        # each listBibl usually contains multiple bibliographic units (biblStruct / bibl)
        children = list(lb.findall("*"))
        if not children:
            # fallback: the listBibl itself may be one entry
            children = [lb]

        for item in children:
            xml_str = ET.tostring(item, encoding="unicode")
            raw_note_elem = _safe_find(item, ".//tei:note[@type='raw_reference']", ns)
            raw_text = _text_of_element(raw_note_elem) if raw_note_elem is not None else _text_of_element(item)
            raw_text = _clean_whitespace(raw_text) or ""

            authors_raw_str = _extract_authors_from_raw(raw_text)
            authors_raw_list = _split_authors_string(authors_raw_str) if authors_raw_str else []

            title_elem = _safe_find(item, ".//tei:title", ns)
            title = _clean_whitespace(_text_of_element(title_elem)) if title_elem is not None else None

            date_elem = _safe_find(item, ".//tei:date", ns)
            year = date_elem.get("when") if date_elem is not None and date_elem.get("when") else (_text_of_element(date_elem) if date_elem is not None else None)

            doi = None
            for idno in _safe_findall(item, ".//tei:idno", ns):
                typ = (idno.get("type") or "").lower()
                txt = _text_of_element(idno)
                if "doi" in typ:
                    doi = txt
                    break
                # common arXiv style / CoRR sometimes stored in idno
                if ("abs/" in txt or "arxiv" in txt.lower() or "corr" in txt.lower()) and doi is None:
                    doi = txt

            # attempt to parse authors from persName and heal them
            pers_list = _parse_persname_elements(item, ns)
            authors_teiled = _heuristic_merge_persnames(pers_list, authors_raw_str)

            # fallback: if TEI produced nothing, use raw list
            if not authors_teiled and authors_raw_list:
                authors_teiled = authors_raw_list

            venue = None
            monogr_title = _safe_find(item, ".//tei:monogr/tei:title", ns)
            if monogr_title is not None:
                venue = _clean_whitespace(_text_of_element(monogr_title))
            else:
                # fallback to idno textual hints
                for idno in _safe_findall(item, ".//tei:idno", ns):
                    t = _text_of_element(idno)
                    if t:
                        venue = t
                        break

            results.append({
                "id": idx,
                "raw": raw_text,
                "xml": xml_str,
                "title": title,
                "year": year,
                "doi": doi,
                "venue": venue,
                "authors_teiled": authors_teiled,
                "authors_raw_str": authors_raw_str,
                "authors_raw_list": authors_raw_list,
            })
            idx += 1
    return results

# ----------------------------
# Full flow: accept path/bytes and call GROBID
# ----------------------------
def find_refs(
    pdf_source: Union[str, bytes, io.BytesIO],
    grobid_url: str = "http://localhost:8070",
    include_raw_citations: bool = True,
    use_fulltext_first: bool = True,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    timeout: int = 120,
) -> List[Dict]:
    """
    Main entry point: accepts:
      - a local file path (str),
      - raw bytes (bytes),
      - io.BytesIO

    Calls GROBID and returns list of parsed references (see parse_references_from_tei_bytes).
    """
    needs_close = False
    # prepare bytes
    if isinstance(pdf_source, str):
        if not os.path.exists(pdf_source):
            raise FileNotFoundError(f"PDF not found: {pdf_source}")
        with open(pdf_source, "rb") as f:
            pdf_bytes = f.read()
    elif isinstance(pdf_source, bytes):
        pdf_bytes = pdf_source
    elif isinstance(pdf_source, io.BytesIO):
        pdf_bytes = pdf_source.getvalue()
    else:
        raise ValueError("pdf_source must be a file path, bytes, or BytesIO")

    data = {}
    if include_raw_citations:
        data['includeRawCitations'] = '1'
    if start_page is not None:
        data['start'] = str(start_page)
    if end_page is not None:
        data['end'] = str(end_page)

    tei_bytes = None
    # Try fulltext first (to capture proper references section), then fallback to processReferences
    if use_fulltext_first:
        try:
            tei_bytes = _call_grobid(grobid_url, "/api/processFulltextDocument", pdf_bytes, data, timeout=timeout)
            refs = parse_references_from_tei_bytes(tei_bytes)
            if refs:
                return refs
        except requests.HTTPError as e:
            # if 204 (no content) or similar, just fall back; otherwise re-raise for visibility
            code = e.response.status_code if e.response is not None else None
            if code and code not in (204, 503):
                raise

    # fallback
    tei_bytes = _call_grobid(grobid_url, "/api/processReferences", pdf_bytes, data, timeout=timeout)
    refs = parse_references_from_tei_bytes(tei_bytes)
    return refs

# ----------------------------
# Quick example when run as script
# ----------------------------
# if __name__ == "__main__":
#     path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1512.05287v5.pdf"
#     print("Parsing references from:", path)
#     refs = find_refs(path)
#     print(f"Found {len(refs)} references. First 5:")
#     for r in refs:
#         print("----"*20)
#         print("id:", r["id"])
#         print("title:", r["title"])
#         print("year:", r["year"])
#         print("venue:", r["venue"])
#         print("doi:", r["doi"])
#         print("authors_teiled:", r["authors_teiled"])
#         print("raw:", r["raw"][:200])
