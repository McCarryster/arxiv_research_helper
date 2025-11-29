import re
from typing import List, Dict, Optional


_YEAR_RE = re.compile(r'(19|20)\d{2}')
_SHORT_CITATION_RE = re.compile(
    r'^\s*([A-Za-z][A-Za-z\-\.\s]*)\s*(?:et\s+al\.?|and)\s*\.?\s*(?:\]?)(?:\D|$)',
    flags=re.IGNORECASE
)


def find_year(text: str) -> Optional[str]:
    m = _YEAR_RE.search(text)
    return m.group(0) if m else None


def normalize_text(text: str) -> str:
    # Lowercase, remove punctuation except spaces, collapse spaces.
    s = re.sub(r'[\[\]\(\)\.,;:"<>]', ' ', text)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s


def extract_surnames_from_short(short: str) -> List[str]:
    """
    From a short citation like "D. Bahdanau et al.2015]" or "Fraser and Marcu2007]",
    attempt to extract one or more surname tokens: ["bahdanau"] or ["fraser","marcu"]
    """
    # Remove trailing bracket(s) and spaces
    short_clean = short.strip().rstrip(']').strip()
    # Try to capture the part before 'et al' or 'and'
    # Examples to handle: "Bahdanau et al.2015]", "Fraser and Marcu2007]"
    # The regex finds the leading name block before 'et al' or 'and'
    m = _SHORT_CITATION_RE.match(short_clean)
    if not m:
        # fallback: try to extract leading word(s) until number-year or bracket
        before_year = re.split(r'\d{4}', short_clean)[0]
        tokens = re.split(r'[,;]| and | & ', before_year)
    else:
        names_block = m.group(1)
        tokens = re.split(r'[,;]| and | & ', names_block)

    surnames = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        # tok may be like "D. Bahdanau" or "Bahdanau"
        parts = tok.split()
        # choose last part as likely surname; remove periods from initials
        surname = parts[-1].replace('.', '')
        # remove non-letter characters
        surname = re.sub(r'[^A-Za-z\-]', '', surname)
        if surname:
            surnames.append(surname.lower())
    return surnames


def is_probably_full_entry(text: str) -> bool:
    """
    Heuristics to classify an entry as a full bibliographic string:
    - contains a year followed by a period and more words (e.g. "2015. Neural machine")
    - or length is reasonably long (> 60)
    - or contains typical tokens like 'In', 'ICLR', 'ACL', 'NIPS', 'EMNLP', 'Computational'
    """
    if _YEAR_RE.search(text):
        # pattern '2015.' followed by a space and a capital letter (title)
        if re.search(r'\b(19|20)\d{2}\.\s+[A-Za-z]', text):
            return True
    keywords = [' in ', 'iclr', 'acl', 'nips', 'emnlp', 'in proceedings', 'computational']
    lowered = text.lower()
    if any(k in lowered for k in keywords):
        return True
    if len(text) > 60:
        return True
    return False


def deduplicate_refs(entries: List[str]) -> List[str]:
    """
    Main function:
      - identify candidate 'full' entries
      - for each short entry attempt to match a full entry by year + surname
      - build result preserving the original order but removing matched short entries
      - deduplicate fully identical (normalized) full entries
    """
    # Precompute full-entry candidates and their normalized forms
    full_candidates = []
    for i, e in enumerate(entries):
        if is_probably_full_entry(e):
            y = find_year(e)
            full_candidates.append({
                'index': i,
                'text': e,
                'year': y,
                'norm': normalize_text(e)
            })

    # Map normalized full texts to keep unique ones (preserve first occurrence order)
    seen_full_norms = set()
    unique_full_texts_by_index: Dict[int, str] = {}
    for f in full_candidates:
        if f['norm'] not in seen_full_norms:
            seen_full_norms.add(f['norm'])
            unique_full_texts_by_index[f['index']] = f['text']
        else:
            # duplicate full entry found; we still allow matching, but will not keep duplicate
            pass

    # For quick search: list of tuples (index, text_lower, year)
    search_list = []
    for idx, txt in unique_full_texts_by_index.items():
        search_list.append((idx, txt, find_year(txt), normalize_text(txt)))

    # Determine which short entries to remove
    keep = [True] * len(entries)

    for i, e in enumerate(entries):
        # Skip those already identified as unique fulls to keep
        if i in unique_full_texts_by_index:
            continue

        # Quick test: does it look like a short citation?
        year = find_year(e)
        lowered = e.lower()
        if not year:
            # no year — unlikely to be a short citation we want to replace
            continue

        # If it's long and already a full entry, skip
        if is_probably_full_entry(e):
            # might be a full entry that just didn't get into unique_full_texts_by_index due to length
            continue

        # If the text contains 'et al' or ' and ' it's likely a short citation
        if ('et al' in lowered) or (' and ' in lowered and len(e) < 80):
            surnames = extract_surnames_from_short(e)
            if not surnames:
                continue

            # Try to find a matching full candidate that contains the year and at least one surname
            found_match = False
            for idx, full_txt, full_year, full_norm in search_list:
                if full_year != year:
                    continue
                # check if any surname appears in the full text (simple substring, case-ins.)
                full_lower = full_txt.lower()
                matching_surname_count = sum(1 for s in surnames if s in full_lower)
                # If at least one surname matches, accept. Prefer matches with more surname hits.
                if matching_surname_count > 0:
                    found_match = True
                    break

            if found_match:
                keep[i] = False  # mark short citation for removal

    # Build resulting list preserving original order, keeping unique fulls (first occurrence)
    result = []
    kept_norms = set()
    for idx, e in enumerate(entries):
        if not keep[idx]:
            # removed short citation
            continue

        # For full entries, avoid keeping duplicates (normalize)
        if is_probably_full_entry(e):
            norm = normalize_text(e)
            if norm in kept_norms:
                continue
            kept_norms.add(norm)
            result.append(e)
        else:
            # not classified as full (or we chose to keep it): include as-is
            result.append(e)

    return result


# # ---------------------------
# # Example usage on your sample:
# # ---------------------------
# if __name__ == "__main__":
#     sample = ['Bahdanau et al.2015', 'D. Bahdanau, K. Cho, and Y. Bengio. 2015. Neural machine translation by jointly learning to align and translate. In ICLR.', 'Buck et al.2014]', 'Christian Buck, Kenneth Heafield, and Bas van Ooyen. 2014. N-gram counts and lan- guage models from the common crawl. In LREC.', 'Cho et al.2014]', 'Kyunghyun Cho, Bart van Merrien- boer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In EMNLP.', 'Fraser and Marcu2007]', 'Alexander Fraser and Daniel Marcu. 2007. Measuring word alignment quality for statistical machine translation. Computational Linguistics, 33(3):293-303.', 'Gregor et al.2015]', 'Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, and Daan Wier- stra. 2015. DRAW: A recurrent neural network for image generation. In ICML.', 'Jean et al.2015]', 'Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. On using very large target vocabulary for neural ma- chine translation. In ACL.', 'Kalchbrenner and Blunsom2013]', 'N. Kalchbrenner and P. Blunsom. 2013. Recurrent continuous translation models. In EMNLP.', 'Koehn et al.2003]', 'Philipp Koehn, Franz Josef Och, and Daniel Marcu. 2003. Statistical phrase-based translation. In NAACL.', 'Liang et al.2006]', 'P. Liang, B. Taskar, and D. Klein. 2006. Alignment by agreement. In NAACL.', 'Luong et al.2015]', 'M.-T. Luong, I. Sutskever, Q. V. Le, O. Vinyals, and W. Zaremba. 2015. Addressing the rare word problem in neural machine translation. In ACL.', 'Mnih et al.2014]', 'Volodymyr Mnih, Nicolas Heess, Alex Graves, and Koray Kavukcuoglu. 2014. Re- current models of visual attention. In NIPS.', 'Papineni et al.2002]', 'Kishore Papineni, Salim Roukos, Todd Ward, and Wei jing Zhu. 2002. Bleu: a method for automatic evaluation of machine trans- lation. In ACL.', 'Sutskever et al.2014]', 'I. Sutskever, O. Vinyals, and Q. V. Le. 2014. Sequence to sequence learning with neural networks. In NIPS.', 'Xu et al.2015]', 'Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron C. Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Ben- gio. 2015. Show, attend and tell: Neural image cap- tion generation with visual attention. In ICML.', 'Zaremba et al.2015]', 'Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. 2015. Recurrent neural network regularization. In ICLR.']

#     cleaned = remove_short_duplicates(sample)

#     for i, d in enumerate(cleaned):
#         print(i+1, d)