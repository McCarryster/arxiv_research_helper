"""
find_arxiv_extended_fixed2.py

Improved heuristic parser for raw references that handles cases like:
  "Vu Pham, Theodore Bluche, Christopher Kermorvant, and Jerome Louradour.
   Dropout improves recurrent neural networks for handwriting recognition.
   In ICFHR. IEEE, 2014."

Function:
    find_arxiv(paper_input, max_results=10, title_threshold=0.87, min_author_overlap=1)

paper_input may be:
    - a dict like your original structure (contains 'title', 'authors_teiled' or 'authors_raw_list', 'year', etc.)
    - OR a raw reference string

Return:
    - a dict with arXiv fields on success (keys: id, title, authors, published, link)
    - False if no acceptable arXiv match found or on error.
"""

import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from difflib import SequenceMatcher
import re
from typing import Dict, Any, List, Union, Optional

# --- Helpers -----------------------------------------------------------------
def normalize_text(s: Optional[str]) -> str:
    """Lowercase, remove extra spaces and punctuation-like differences."""
    if s is None:
        return ""
    s = s.replace("\n", " ")
    # collapse hyphenation that came from line breaks like "Ben- gio"
    s = re.sub(r"-\s+", "", s)
    # replace most punctuation with spaces (keeps alphanumerics)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def surname(name: str) -> str:
    """Return the last token of a person's name (basic surname heuristic)."""
    if not name:
        return ""
    name = name.replace("\n", " ").strip()
    parts = [p for p in re.split(r"\s+", name) if p]
    return parts[-1].lower() if parts else ""

def title_similarity(a: str, b: str) -> float:
    """Return SequenceMatcher ratio of normalized titles."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def extract_entry_authors(entry_elem: ET.Element, ns: Dict[str,str]) -> List[str]:
    """Given an <entry> element from arXiv/Atom, return list of author names."""
    authors = []
    for author in entry_elem.findall("atom:author", ns):
        name_elem = author.find("atom:name", ns)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())
    return authors

def entry_to_dict(entry_elem: ET.Element, ns: Dict[str,str]) -> Dict[str, Any]:
    """Extract fields from an arXiv Atom <entry> element into a dict."""
    entry_id = entry_elem.find("atom:id", ns)
    entry_title = entry_elem.find("atom:title", ns)
    entry_published = entry_elem.find("atom:published", ns)
    entry_link = None
    # find link with rel="alternate" or first link
    for link in entry_elem.findall("atom:link", ns):
        if link.get("rel") == "alternate":
            entry_link = link.get("href")
            break
        if entry_link is None:
            entry_link = link.get("href")
    authors = extract_entry_authors(entry_elem, ns)
    return {
        "id": (entry_id.text.strip() if entry_id is not None and entry_id.text else None),
        "title": (entry_title.text.strip() if entry_title is not None and entry_title.text else None),
        "authors": authors,
        "published": (entry_published.text.strip() if entry_published is not None and entry_published.text else None),
        "link": entry_link
    }

# --- Raw reference parsing (more robust) ------------------------------------
def looks_like_venue_fragment(s: str) -> bool:
    """Return True if s looks like a venue/publisher fragment (short, starts with 'In', or contains known publisher tokens)."""
    if not s:
        return True
    s_clean = s.strip()
    if len(s_clean.split()) <= 2:
        return True
    if re.match(r'^(in|proc|proceedings|ieee|acm|springer|elsevier|cambridge|oxford)\b', s_clean, flags=re.IGNORECASE):
        return True
    # if it contains only capitalized acronyms or very short tokens (e.g., "ICFHR")
    if re.fullmatch(r'([A-Z]{2,6}(\s*[,:])?)+', s_clean):
        return True
    return False

def parse_raw_reference(raw: str) -> Dict[str, Any]:
    """
    Heuristic parser for a raw reference string.
    Returns a dict with keys: 'raw', 'title', 'authors_teiled', 'authors_raw_list', 'year'
    (fields may be empty if not found).
    """
    if not isinstance(raw, str):
        return {"raw": "", "title": "", "authors_teiled": [], "authors_raw_list": [], "year": None}

    # Pre-clean: remove newline hyphenation, collapse spaces
    r = raw.replace("\n", " ")
    r = re.sub(r"-\s+", "", r)     # fix "Ben- gio" -> "Bengio"
    r = re.sub(r"\s+", " ", r).strip()

    # Split into sentences by dot/semicolon/colon patterns
    sentences = [s.strip().rstrip('.') for s in re.split(r'\.\s+|\;\s+|\:\s+', r) if s.strip()]

    # Try to find a 4-digit year between 1900-2099 in the whole string
    year_match = re.search(r"\b(19|20)\d{2}\b", r)
    year = year_match.group(0) if year_match else None

    authors: List[str] = []
    title = ""

    # If the year appears inside one of the sentences, prefer a sentence earlier than that one that looks like a title
    year_sentence_idx = None
    if year:
        for i, s in enumerate(sentences):
            if re.search(r"\b" + re.escape(year) + r"\b", s):
                year_sentence_idx = i
                break

    if year_sentence_idx is not None and year_sentence_idx > 0:
        # Search backwards from the year sentence for the first prior sentence that is not a short/venue fragment
        chosen_title = None
        for j in range(year_sentence_idx - 1, -1, -1):
            cand = sentences[j].strip()
            if not cand:
                continue
            # Skip if looks like a venue/publisher fragment or too short
            if looks_like_venue_fragment(cand):
                continue
            # Prefer sentences with at least 3 words (simple heuristic for title)
            if len(cand.split()) >= 3:
                chosen_title = cand
                break
        # If none found by that strict rule, fall back to the immediate previous sentence (even if short)
        if chosen_title is None:
            chosen_title = sentences[year_sentence_idx - 1].strip()
        title = chosen_title

        # Authors are typically the first sentence if it's not the title
        if sentences:
            first = sentences[0].strip()
            if first and first != title:
                # parse author names by commas/and
                raw_authors = re.split(r',\s*and\s+|,\s+| and ', first)
                raw_authors = [a.strip().strip('.') for a in raw_authors if a.strip()]
                authors = raw_authors
            else:
                # fallback: try second sentence if different
                if len(sentences) > 1 and sentences[1].strip() != title:
                    raw_authors = re.split(r',\s*and\s+|,\s+| and ', sentences[1])
                    raw_authors = [a.strip().strip('.') for a in raw_authors if a.strip()]
                    authors = raw_authors
    else:
        # Fallback older heuristics when year missing or when year cannot be isolated into sentences
        if year_match:
            # assume authors before year, title after year
            authors_segment = r[:year_match.start()].strip()
            authors_segment = authors_segment.rstrip(". ,;:")
            if authors_segment:
                raw_authors = re.split(r',\s*and\s+|,\s+| and ', authors_segment)
                raw_authors = [a.strip().strip('.') for a in raw_authors if a.strip()]
                authors = raw_authors
            # title candidate: text after year up to next period or venue marker
            tail = r[year_match.end():].lstrip(" .")
            m_in = re.search(r'\bIn\b|Proceedings|Proc\.|Conference|ICLR|NIPS|NeurIPS|ICFHR|IEEE|ACM', tail, flags=re.IGNORECASE)
            if m_in:
                title_candidate = tail[:m_in.start()].strip().rstrip(".")
            else:
                m = re.search(r'(.+?)(?:\.\s|$)', tail)
                title_candidate = m.group(1).strip() if m else tail.strip()
            title = title_candidate
        else:
            # No year: use sentence heuristics: prefer longest sentence not likely to be authors
            if sentences:
                if len(sentences) == 1:
                    maybe = sentences[0]
                    # if many commas and short tokens, likely authors
                    tokens = maybe.split()
                    if maybe.count(',') >= 1 and len(tokens) <= 20:
                        raw_authors = re.split(r',\s*and\s+|,\s+| and ', maybe)
                        raw_authors = [a.strip().strip('.') for a in raw_authors if a.strip()]
                        authors = raw_authors
                        title = ""
                    else:
                        title = maybe
                        authors = []
                else:
                    # try to find venue index then take previous as title
                    vidx = None
                    for i, s in enumerate(sentences):
                        if re.search(r'\bProceedings\b|\bConference\b|\bICLR\b|\bNIPS\b|\bNeurIPS\b|\bICFHR\b|\bIEEE\b|\bArXiv\b|\bJournal\b|\bProc\.', s, flags=re.IGNORECASE):
                            vidx = i
                            break
                    if vidx is not None and vidx > 0:
                        # find a proper title before vidx by the same logic (skip short fragments)
                        chosen_title = None
                        for j in range(vidx - 1, -1, -1):
                            cand = sentences[j].strip()
                            if not cand:
                                continue
                            if looks_like_venue_fragment(cand):
                                continue
                            if len(cand.split()) >= 3:
                                chosen_title = cand
                                break
                        if chosen_title is None:
                            chosen_title = sentences[vidx - 1].strip()
                        title = chosen_title
                        # authors often in first sentence
                        if sentences[0].strip() != title:
                            raw_authors = re.split(r',\s*and\s+|,\s+| and ', sentences[0])
                            raw_authors = [a.strip().strip('.') for a in raw_authors if a.strip()]
                            authors = raw_authors
                    else:
                        # no venue marker: choose the longest sentence as title, first sentence as authors (if different)
                        sentences_sorted = sorted(sentences, key=lambda x: len(x), reverse=True)
                        title = sentences_sorted[0]
                        if sentences[0] != title:
                            maybe_auths = re.split(r',\s*and\s+|,\s+| and ', sentences[0])
                            maybe_auths = [a.strip().strip('.') for a in maybe_auths if len(a.split()) <= 6]
                            authors = maybe_auths

    # final sanitization
    title = (title or "").strip()
    # If title is suspiciously short (<3 words), try to extend by appending the next sentence if possible
    if title and len(title.split()) < 3:
        try:
            idx = sentences.index(title)
            if idx + 1 < len(sentences):
                title = (title + " " + sentences[idx + 1]).strip()
        except Exception:
            pass

    # sanitize authors list (remove empty and stray punctuation)
    authors = [re.sub(r'[^\w\s\.\-]', '', a).strip() for a in authors if a and a.strip()]

    return {
        "raw": r,
        "title": title,
        "authors_teiled": authors,
        "authors_raw_list": authors,
        "year": year
    }

# --- Main function -----------------------------------------------------------
def find_arxiv(
    paper_input: Union[Dict[str, Any], str],
    max_results: int = 10,
    title_threshold: float = 0.87,
    min_author_overlap: int = 1,
) -> Union[Dict[str, Any], bool]:
    """
    Try to find an arXiv entry matching the given paper metadata or raw reference string.

    Parameters:
        paper_input: dict (your original structure) or a raw reference string
        max_results: how many arXiv results to fetch per query
        title_threshold: minimum SequenceMatcher ratio for title acceptance (0..1)
        min_author_overlap: minimum number of overlapping author surnames required for borderline matches

    Returns:
        dict with keys: 'id', 'title', 'authors', 'published', 'link' on success,
        or False if nothing acceptable is found or on error.
    """
    # Normalize input: if string, parse it
    if isinstance(paper_input, str):
        paper = parse_raw_reference(paper_input)
    elif isinstance(paper_input, dict):
        # make a shallow copy to avoid mutating caller's dict
        paper = dict(paper_input)
    else:
        # unknown input type
        return False

    base_api = "http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={maxr}"
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    # Extract candidate metadata
    title = (paper.get("title") or paper.get("raw") or "")
    authors_list = paper.get("authors_teiled") or paper.get("authors_raw_list") or []
    year = paper.get("year")

    # Prepare surnames from provided authors
    provided_surnames = [surname(a) for a in authors_list if a]
    provided_surnames = [s for s in provided_surnames if s]

    # Build a list of query strings to try, in order of preference
    queries: List[str] = []

    # 1) Exact title search using ti: (quoted) — only if title present
    if title:
        q1 = f'ti:"{title}"'
        queries.append(q1)

    # 2) All words from title (all:) - fallback
    title_words = " ".join(re.findall(r"\w+", title)) if title else ""
    if title_words:
        q2 = f'all:"{title_words}"'
        queries.append(q2)

    # 3) First author + title words
    if provided_surnames and title_words:
        first_author = provided_surnames[0]
        q3 = f'au:{first_author}+AND+all:"{title_words}"'
        queries.append(q3)

    # 4) Author-only search (first author) - broad
    if provided_surnames:
        q4 = f'au:{provided_surnames[0]}'
        queries.append(q4)

    # 5) If nothing constructed above (no title, no authors) — search the entire raw string
    if not queries:
        raw_query = " ".join(re.findall(r"\w+", paper.get("raw", "")))
        if raw_query:
            q5 = f'all:"{raw_query}"'
            queries.append(q5)

    # Iterate queries
    try:
        for q in queries:
            encoded = quote_plus(q, safe=':"')  # keep quotes for arXiv phrase search
            url = base_api.format(query=encoded, maxr=max_results)
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200 or not resp.text:
                continue
            # parse XML
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ns)
            candidates: List[Dict[str, Any]] = []
            for e in entries:
                ed = entry_to_dict(e, ns)
                candidates.append(ed)

            # Evaluate candidates by scoring: title similarity + author overlap
            for cand in candidates:
                cand_title = cand.get("title") or ""
                sim = title_similarity(title, cand_title) if title else 0.0
                # compute author surname overlap
                cand_surnames = [surname(a) for a in cand.get("authors", [])]
                overlap = len(set(s for s in provided_surnames if s and s in cand_surnames))
                # optional year check (if provided) - not strict, just for score hint
                cand_year = None
                if cand.get("published"):
                    try:
                        cand_year = int(cand["published"][:4])
                    except Exception:
                        cand_year = None
                # Decide acceptance:
                # - strong title match (>= title_threshold) -> accept
                # - or somewhat lower title match but author overlap >= min_author_overlap -> accept
                if (title and sim >= title_threshold) or (title and sim >= (title_threshold - 0.12) and overlap >= min_author_overlap) or (not title and overlap >= max(1, min_author_overlap)):
                    # Extra sanity: if provided year exists and candidate year exists, prefer matches within +/-4 years
                    if year and cand_year:
                        try:
                            py = int(year)
                            if abs(py - cand_year) > 4:
                                # too far apart; skip this candidate
                                continue
                        except Exception:
                            pass
                    # Good match found
                    return cand

        # no acceptable candidate found
        return False

    except Exception:
        # network error, XML parse error, etc. — return False per requirement
        return False

# --- Example usage ----------------------------------------------------------
# if __name__ == "__main__":
#     # Example input (the dict you posted)
#     # example = {
#     #     'id': 3,
#     #     'raw': 'Kyunghyun Cho, Bart van Merrienboer, Caglar Gul- cehre, Dzmitry Bahdanau, Fethi Bougares, Hol- ger Schwenk, and Yoshua Bengio. 2014. Learn- ing Phrase Representations using RNN Encoder- Decoder for Statistical Machine Translation. In Pro- ceedings of the 2014 Conference on Empirical Meth- ods in Natural Language Processing (EMNLP), pages 1724-1734, Doha, Qatar. Association for Computational Linguistics.',
#     #     'title': 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation',
#     #     'year': '2014',
#     #     'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merrienboer', 'Caglar Gulcehre', 'Dzmitry Bahdanau', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio'],
#     #     'authors_raw_list': ['Kyunghyun Cho', 'Bart van Merrienboer', 'Caglar Gul- cehre', 'Dzmitry Bahdanau', 'Fethi Bougares', 'Hol- ger Schwenk', 'Yoshua Bengio']
#     # }
#     example = "Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Ben- gio. 2015. Neural Machine Translation by Jointly Learning to Align and Translate. In Proceedings of the International Conference on Learning Represen- tations (ICLR)."
#     found = find_arxiv(example)
#     if not found:
#         print(False)
#     else:
#         print("Match found:")
#         print("  arXiv id/url:", found.get("id") or found.get("link")) # type: ignore
#         print("  title:", found.get("title")) # type: ignore
#         print("  authors:", found.get("authors")) # type: ignore
#         print("  published:", found.get("published")) # type: ignore