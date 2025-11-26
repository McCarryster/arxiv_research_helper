from typing import List, Dict, Optional, Set
import requests
import xml.etree.ElementTree as ET

# Default GROBID server
GROBID_BASE = "http://localhost:8070"

def call_grobid_process_citation_list_single(citation: str, host: str = GROBID_BASE) -> str:
    """
    Call GROBID /api/processCitationList for a single citation.
    Returns the TEI XML string.
    """
    url = f"{host.rstrip('/')}/api/processCitationList"
    headers = {"Accept": "application/xml"}
    data = {"citations": citation}
    resp = requests.post(url, data=data, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text

def strip_ns(tag: str) -> str:
    """Return local-name of an XML tag (strip namespace)."""
    return tag.split("}")[-1] if tag and "}" in tag else tag or ""

def assemble_person_name_from_pers(pers_el: ET.Element) -> str:
    """Build full name from <persName>."""
    parts = []
    for child in pers_el:
        local = strip_ns(child.tag)
        if local in ("forename", "initial", "forenameInitial", "given"):
            if child.text and child.text.strip():
                parts.append(child.text.strip())
        elif local in ("surname", "family"):
            if child.text and child.text.strip():
                parts.append(child.text.strip())
    if not parts:
        text = (pers_el.text or "").strip()
        if text:
            parts = [text]
    return " ".join(parts).strip()

def assemble_name_from_author_el(author_el: ET.Element) -> str:
    """Fallback name assembly from <author>."""
    for child in author_el:
        if strip_ns(child.tag) == "persName":
            return assemble_person_name_from_pers(child)
    surname = None
    forename_parts = []
    for child in author_el:
        local = strip_ns(child.tag)
        if local in ("surname", "family"):
            surname = (child.text or "").strip()
        elif local in ("forename", "given", "initial"):
            if child.text and child.text.strip():
                forename_parts.append(child.text.strip())
    if surname or forename_parts:
        return " ".join(forename_parts + ([surname] if surname else [])).strip()
    raw = "".join((c.text or "") for c in author_el).strip()
    if raw:
        return raw
    if author_el.text and author_el.text.strip():
        return author_el.text.strip()
    return ""

def parse_grobid_tei(tei_xml: str, original_citation: Optional[str] = None) -> List[Dict]:
    """
    Parse TEI XML returned by GROBID into a list of dicts:
    [{'title': ..., 'authors_teiled': [...], 'original_citation': ...}, ...]
    """
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    results = []
    bibl_structs = [el for el in root.iter() if strip_ns(el.tag) == "biblStruct"]
    if not bibl_structs:
        for el in root.iter():
            if strip_ns(el.tag) in ("listBibl", "bibl"):
                for child in el.iter():
                    if strip_ns(child.tag) == "biblStruct":
                        bibl_structs.append(child)

    for bibl in bibl_structs:
        title_text = ""
        authors_list: List[str] = []

        # Extract title
        for t in bibl.iter():
            if strip_ns(t.tag) == "title" and (t.text or "").strip():
                title_text = t.text.strip()  # type: ignore
                break

        # Extract authors
        for p in bibl.iter():
            if strip_ns(p.tag) == "persName":
                name = assemble_person_name_from_pers(p)
                if name and name not in authors_list:
                    authors_list.append(name)

        if not authors_list:
            for a in bibl.iter():
                if strip_ns(a.tag) == "author":
                    name = assemble_name_from_author_el(a)
                    if name and name not in authors_list:
                        authors_list.append(name)

        cleaned = []
        for a in authors_list:
            if a and a not in cleaned:
                cleaned.append(a)

        results.append({
            "title": title_text,
            "authors_teiled": cleaned,
            "original_citation": original_citation or ""
        })
    return results

def parse_mark_refs(citation_list: Set[str], host: str = GROBID_BASE) -> List[Dict]:
    """
    Parse a list of citation strings via GROBID.
    Returns [{'title': ..., 'authors_teiled': [...], 'original_citation': ...}, ...].
    """
    all_results: List[Dict] = []
    for citation in citation_list:
        try:
            tei = call_grobid_process_citation_list_single(citation, host)
            parsed = parse_grobid_tei(tei, original_citation=citation)
            if parsed:
                all_results.extend(parsed)
            else:
                all_results.append({
                    "title": "",
                    "authors_teiled": [],
                    "original_citation": citation,
                    "error": "No biblStruct parsed"
                })
        except requests.exceptions.RequestException as e:
            all_results.append({
                "title": "",
                "authors_teiled": [],
                "original_citation": citation,
                "error": str(e)
            })
    return all_results

# # Optional: quick test when run directly
# if __name__ == "__main__":
#     test_citations = [
#         "[36] S. E. Robertson and S. Walker. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. pages 232–241. Springer-Verlag New York, Inc., 1994.",
#         "[15] S. P. Harter. A probabilistic approach to automatic keyword indexing. JASIS, 26(5):280–289, 1975."
#     ]
#     for item in parse_mark_refs(test_citations):
#         print({"authors_teiled": item["authors_teiled"], "title": item["title"]})