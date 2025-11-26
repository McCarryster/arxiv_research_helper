
import re
from typing import Tuple, Dict, Any, Optional
import requests
from lxml import etree # type: ignore

def uses_numbered_citations(pdf_path: str,
                           grobid_url: str = "http://localhost:8070",
                           timeout: int = 120,
                           min_bib_entries_to_trust: int = 5,
                           bib_majority_ratio: float = 0.6,
                           return_report: bool = False
                           ) -> Any:
    """
    Args:
        pdf_path: path to local PDF file.
        grobid_url: base URL of running GROBID service (e.g. "http://localhost:8070").
        timeout: HTTP timeout.
        min_bib_entries_to_trust: minimum number of bibliography entries required
                                  to trust bibliography-based majority decision.
        bib_majority_ratio: fraction required for majority in bibliography (e.g. 0.6 means 60%).
        return_report: if True, return (bool, report_dict). If False, return bool.

    Returns:
        bool OR (bool, report_dict) if return_report True.

    Raises:
        requests.RequestException for network errors.
        ValueError if GROBID output cannot be parsed.
    """

    # --- Helpers / regexes ---
    bracket_num_re = re.compile(r"\[\s*\d+(?:\s*(?:,|;|:|\-|–|—)\s*\d+)*(?:\s*(?:\-|–|—)\s*\d+)?\s*\]")
    paren_num_re = re.compile(r"\(\s*\d+(?:\s*(?:,|;|:|\-|–|—)\s*\d+)*(?:\s*(?:\-|–|—)\s*\d+)?\s*\)")
    # matches patterns like (Luong et al., 2015) or (Smith, 2019)
    author_year_intext_re = re.compile(
        r"\([A-Z][A-Za-z\-\s\.]+(?:et al\.|and [A-Z][A-Za-z\-]+)?,?\s*(?:19|20)\d{2}\)"  # (Author et al., 2015)
        r"|\b[A-Z][A-Za-z\-\.\s]+et al\.\s*\(\s*(?:19|20)\d{2}\s*\)"  # Author et al. (2015)
        r"|\b[A-Z][A-Za-z\-\.\s]+\,\s*(?:19|20)\d{2}\b"  # "Author, 2015" in references lines
    )
    year_re = re.compile(r"\b(?:19|20)\d{2}\b")
    bib_bracket_label_re = re.compile(r"^\s*\[\s*\d+\s*\]")
    bib_leading_num_re = re.compile(r"^\s*(?:\d{1,4}\s*[\.\)]\s+|\d{1,4}\s+-\s+)")  # "1." "1)""1 -"
    # superscript digits (approx): include common unicode superscript digits
    superscript_chars = ''.join([
        '\u00B9', '\u00B2', '\u00B3',
        *[chr(c) for c in range(0x2070, 0x207A)]
    ])
    superscript_re = re.compile(rf"[{re.escape(superscript_chars)}]")

    def element_text(el):
        if el is None:
            return ""
        return " ".join([t for t in el.itertext() if t and t.strip()]).strip()

    # --- 1) Call GROBID processFulltextDocument ---
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    params = {"consolidateCitations": "0"}  # keep simple
    with open(pdf_path, "rb") as fd:
        files = {"input": (pdf_path, fd, "application/pdf")}
        resp = requests.post(endpoint, files=files, params=params, timeout=timeout)
        resp.raise_for_status()
        tei_xml = resp.content

    # --- 2) Parse TEI XML ---
    try:
        root = etree.fromstring(tei_xml)
    except Exception as e:
        raise ValueError(f"Failed to parse TEI XML from GROBID: {e}")

    NS_TEI = "http://www.tei-c.org/ns/1.0"
    ns = {"tei": NS_TEI}

    report: Dict[str, Any] = {
        "bib_count": 0,
        "bib_numeric_label_count": 0,
        "bib_author_year_count": 0,
        "sections_examined": {},
        "body_counts": {},
        "final_decision": None,
        "reason": None
    }

    # --- 3) Extract bibliography entries robustly (preferred signal) ---
    bib_entries_texts = []

    # candidate XPaths for bibliography containers (covers common TEI produced by GROBID)
    bib_containers = [
        ".//tei:listBibl",
        ".//tei:div[@type='references']",
        ".//tei:back//tei:listBibl",
        ".//tei:back//tei:div[@type='references']",
        ".//tei:note[@type='references']",
        ".//tei:biblStruct"
    ]

    for xp in bib_containers:
        for container in root.findall(xp, namespaces=ns):
            # try to get child bibl items
            items = container.findall(".//tei:biblStruct", namespaces=ns) or \
                    container.findall(".//tei:bibl", namespaces=ns) or \
                    container.findall(".//tei:listItem", namespaces=ns) or \
                    container.findall(".//tei:li", namespaces=ns)
            if items:
                for it in items:
                    txt = element_text(it)
                    if txt:
                        bib_entries_texts.append(txt)
            else:
                # maybe it's a paragraph-per-reference container
                ps = container.findall(".//tei:p", namespaces=ns)
                if ps:
                    for p in ps:
                        t = element_text(p)
                        if t:
                            bib_entries_texts.append(t)
                else:
                    # fallback: whole container text split heuristically
                    full = element_text(container)
                    if full:
                        parts = re.split(r"\n{2,}|\r{2,}|\n\s*\d+\.\s+", full)
                        for p in parts:
                            p = p.strip()
                            if len(p) > 20:
                                bib_entries_texts.append(p)

    # remove duplicates/preamble noise
    seen = set()
    clean_bib_entries = []
    for t in bib_entries_texts:
        t2 = re.sub(r"\s+", " ", t).strip()
        if t2 and t2 not in seen:
            clean_bib_entries.append(t2)
            seen.add(t2)

    # if none found, try looser search for any biblStruct in document
    if not clean_bib_entries:
        for b in root.findall(".//tei:biblStruct", namespaces=ns):
            t = element_text(b)
            if t and len(t) > 20:
                t2 = re.sub(r"\s+", " ", t).strip()
                if t2 not in seen:
                    clean_bib_entries.append(t2)
                    seen.add(t2)

    # populate bib counts
    report["bib_count"] = len(clean_bib_entries)

    for entry in clean_bib_entries:
        # numeric label at start like "[1]" or "1. " or "1) "
        if bib_bracket_label_re.search(entry) or bib_leading_num_re.match(entry):
            report["bib_numeric_label_count"] += 1
            continue
        # author-year style in bibliography entry: look for year + author token
        if year_re.search(entry) and re.search(r"[A-Z][a-z]{2,}\s+(and|&|,|\.)", entry):
            report["bib_author_year_count"] += 1
            continue
        # fallback: "et al." + year indicates author-year
        if "et al." in entry and year_re.search(entry):
            report["bib_author_year_count"] += 1
            continue
        # otherwise ignore ambiguous entries

    # --- 4) If bibliography is large enough, trust majority there ---
    if report["bib_count"] >= min_bib_entries_to_trust:
        num = report["bib_numeric_label_count"]
        auth = report["bib_author_year_count"]
        # decide by majority ratio
        if num >= max(2, int(bib_majority_ratio * report["bib_count"])):
            report["final_decision"] = True
            report["reason"] = "bibliography_majority_numeric"
            return (True, report) if return_report else True
        if auth >= max(2, int(bib_majority_ratio * report["bib_count"])):
            report["final_decision"] = False
            report["reason"] = "bibliography_majority_author_year"
            return (False, report) if return_report else False
        # No clear majority in bibliography: fall through to section inspection

    # --- 5) Inspect key sections (where citations usually appear) ---
    # Keywords to find likely sections that mention citations: Introduction, Related, Background, Method, Experiments
    section_keywords = [
        "introduction", "related", "related work", "background", "method", "methods",
        "approach", "experiments", "evaluation", "results"
    ]
    # find all div/section elements with heads and examine those matching keywords
    sections_to_check = []
    for div in root.findall(".//tei:div", namespaces=ns):
        head = div.find(".//tei:head", namespaces=ns)
        if head is not None:
            head_txt = element_text(head).lower()
            for kw in section_keywords:
                if kw in head_txt:
                    sections_to_check.append((head_txt, div))
                    break

    # if none matched heuristically, pick first few top-level sections (defensive fallback)
    if not sections_to_check:
        # choose first up to 3 divs with a head
        count = 0
        for div in root.findall(".//tei:div", namespaces=ns):
            head = div.find(".//tei:head", namespaces=ns)
            if head is not None:
                sections_to_check.append((element_text(head).lower(), div))
                count += 1
            if count >= 3:
                break

    # If still none found, we'll inspect the whole body later
    section_cumulative = {
        "bracket_intext": 0,
        "paren_intext": 0,
        "author_year_intext": 0,
        "superscript_intext": 0,
        "raw_numeric_tokens": 0,
        "sections_counted": 0
    }

    for head_txt, div in sections_to_check:
        sec_txt = element_text(div)
        bcount = len(bracket_num_re.findall(sec_txt))
        pcount = len(paren_num_re.findall(sec_txt))
        aycount = len(author_year_intext_re.findall(sec_txt))
        sscount = len(superscript_re.findall(sec_txt))
        raw_nums = len(re.findall(r"\b\d{1,3}\b", sec_txt))
        section_cumulative["bracket_intext"] += bcount
        section_cumulative["paren_intext"] += pcount
        section_cumulative["author_year_intext"] += aycount
        section_cumulative["superscript_intext"] += sscount
        section_cumulative["raw_numeric_tokens"] += raw_nums
        section_cumulative["sections_counted"] += 1
        report["sections_examined"][head_txt if head_txt else f"section_{len(report['sections_examined'])+1}"] = {
            "bracket_intext": bcount,
            "paren_intext": pcount,
            "author_year_intext": aycount,
            "superscript_intext": sscount,
            "raw_numeric_tokens": raw_nums
        }

    # --- 6) Triage decision based on section signals (strict thresholds) ---
    # thresholds here are intentionally strict to avoid false positives on author-year papers
    # e.g. require multiple bracket occurrences in key sections to conclude numbered style
    b_in = section_cumulative["bracket_intext"]
    p_in = section_cumulative["paren_intext"]
    ay_in = section_cumulative["author_year_intext"]
    ss_in = section_cumulative["superscript_intext"]
    raw_in = section_cumulative["raw_numeric_tokens"]

    report["body_counts"] = {
        "section_bracket_count": b_in,
        "section_paren_count": p_in,
        "section_author_year_count": ay_in,
        "section_superscript_count": ss_in,
        "section_raw_numeric_tokens": raw_in,
        "sections_examined": section_cumulative["sections_counted"]
    }

    # Rule set (conservative):
    # - Strong author-year evidence in sections -> non-numbered
    if ay_in >= 3 and (b_in + ss_in) <= 1:
        report["final_decision"] = False
        report["reason"] = "sections_author_year_strong"
        return (False, report) if return_report else False

    # - Several bracket occurrences in key sections strongly indicate numbered style
    if b_in >= 4 or (b_in >= 2 and ss_in >= 2):
        report["final_decision"] = True
        report["reason"] = "sections_bracket_strong"
        return (True, report) if return_report else True

    # - Strong superscript-only evidence
    if ss_in >= 4:
        report["final_decision"] = True
        report["reason"] = "sections_superscript_strong"
        return (True, report) if return_report else True

    # - If author-year occurrences dominate and are >=4 -> non-numbered
    if ay_in >= 4 and ay_in > (b_in + ss_in):
        report["final_decision"] = False
        report["reason"] = "sections_author_year_dominate"
        return (False, report) if return_report else False

    # --- 7) Final fallback: scan whole body text but remain conservative ---
    body_el = root.find(".//tei:text/tei:body", namespaces=ns)
    full_body = element_text(body_el) if body_el is not None else element_text(root)

    total_bracket = len(bracket_num_re.findall(full_body))
    total_paren_num = len(paren_num_re.findall(full_body))
    total_ay = len(author_year_intext_re.findall(full_body))
    total_sup = len(superscript_re.findall(full_body))

    report["body_counts"].update({
        "total_bracket": total_bracket,
        "total_paren_num": total_paren_num,
        "total_author_year_intext": total_ay,
        "total_superscript": total_sup
    })

    # fallback rules (very conservative):
    if total_bracket >= 8 or (total_bracket >= 5 and total_sup >= 3):
        report["final_decision"] = True
        report["reason"] = "body_bracket_fallback"
        return (True, report) if return_report else True

    if total_ay >= 8 and total_ay > (total_bracket + total_sup):
        report["final_decision"] = False
        report["reason"] = "body_author_year_fallback"
        return (False, report) if return_report else False

    # If nothing decisive, default to non-numbered to avoid false positives.
    report["final_decision"] = False
    report["reason"] = "conservative_default_no_clear_numeric_evidence"
    return (False, report) if return_report else False



# if __name__ == "__main__":

#     pdf_paths = [
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1308.0850v5.pdf", # markers
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1508.04025v5.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1508.07909v5.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1511.06114v4.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1601.06733v7.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1602.02410v2.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1607.06450v1.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1608.05859v3.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1609.08144v2.pdf", # markers
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1610.02357v3.pdf", # markers
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1610.10099v2.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1701.06538v1.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1703.03130v1.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1703.10722v3.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1705.03122v3.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1705.04304v3.pdf",
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1706.03762v7.pdf", # markers
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1512.05287v5.pdf", # markers
#         "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_pdfs/1602.01137v1.pdf", # markers
#     ]

#     grobid = "http://localhost:8070"
#     for path in pdf_paths:
#         print("Uses numbered citations?:", uses_numbered_citations(path, grobid_url=grobid))

#     # try:
#     #     print("Uses numbered citations?:", uses_numbered_citations(pdf, grobid_url=grobid))
#     # except Exception as e:
#     #     print("Error:", e)