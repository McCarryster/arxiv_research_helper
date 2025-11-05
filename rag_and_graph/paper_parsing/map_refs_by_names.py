"""
Print only matched authors from a provided author-list against a text.

Just paste and run. No CLI/argparse used.

Example output (for the provided example):
M.-T Luong
Sébastien Jean
"""

from typing import List, Dict
import unicodedata
import re
import difflib


def _normalize(s: str) -> str:
    """Lowercase, remove diacritics, collapse whitespace and non-word chars (keeps hyphen/apostrophe)."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^0-9a-zA-Z\-\']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_name_tokens(name: str) -> List[str]:
    """Split a name into tokens (keeps hyphenated pieces)."""
    name = re.sub(r"[,()]", " ", name)
    tokens = re.split(r"[^0-9A-Za-z'\-]+", name)
    tokens = [t for t in tokens if t and t.strip()]
    return tokens


def _is_initial(token: str) -> bool:
    """Heuristic to tell if a token is an initial (e.g. 'D', 'D.', 'M.-T')."""
    if not token:
        return False
    t = token.replace(".", "")
    if re.fullmatch(r"(?:[A-Za-z](?:-[A-Za-z])*)", t) and all(len(part) == 1 for part in t.split("-")):
        return True
    if len(t) == 1:
        return True
    return False


def _surname_candidates_from_author(raw_name: str) -> List[str]:
    """
    Produce likely surname candidates from an author string:
      - last non-initial token
      - first non-initial token (handles reversed "Surname Given")
      - last token (fallback)
      - last two tokens joined (compound surname)
    Returns normalized candidates.
    """
    tokens = _split_name_tokens(raw_name)
    if not tokens:
        return []
    non_initial_tokens = [t for t in tokens if not _is_initial(t)]
    candidates = []
    if non_initial_tokens:
        candidates.append(non_initial_tokens[-1])  # last non-initial
        if non_initial_tokens[0] != non_initial_tokens[-1]:
            candidates.append(non_initial_tokens[0])  # first non-initial (reversed)
    if tokens[-1] not in candidates:
        candidates.append(tokens[-1])
    if len(tokens) >= 2:
        last_two = f"{tokens[-2]} {tokens[-1]}"
        if last_two not in candidates:
            candidates.append(last_two)
    # normalize and deduplicate in order
    seen = set()
    out = []
    for c in candidates:
        cn = _normalize(c)
        if cn and cn not in seen:
            seen.add(cn)
            out.append(cn)
    return out


def _generate_given_tokens(raw_name: str) -> List[str]:
    """Return normalized given-name tokens (non-initials excluding chosen surname) for proximity checks."""
    tokens = _split_name_tokens(raw_name)
    non_initial = [t for t in tokens if not _is_initial(t)]
    given = []
    if non_initial:
        surname = non_initial[-1]
        for t in tokens:
            if _normalize(t) == _normalize(surname):
                continue
            if not _is_initial(t):
                given.append(t)
    else:
        given = [t for t in tokens if not _is_initial(t)]
    return [_normalize(g) for g in given if _normalize(g)]


def _word_list_with_positions(norm_text: str) -> List[tuple]:
    """Return list of (word, index) from normalized text."""
    words = re.findall(r"[0-9a-zA-Z'\-]+", norm_text)
    return [(w, i) for i, w in enumerate(words)]


def find_authors_in_text(authors: List[str], text: str, fuzzy_threshold: float = 0.92) -> Dict[str, dict]:
    """
    Check presence of each author in authors within text.

    Returns:
      dict mapping original author string -> match-info dict containing 'present' bool and 'method'.
    """
    norm_text = _normalize(text)
    results = {}
    words_with_pos = _word_list_with_positions(norm_text)
    words = [w for w, _ in words_with_pos]

    for author in authors:
        surname_candidates = _surname_candidates_from_author(author)
        given_tokens = _generate_given_tokens(author)
        result = {"present": False, "method": None, "details": {"candidates": surname_candidates}}
        matched = False

        # 1) Citation-like patterns: "<surname> et al." or "<surname> (YYYY)" etc.
        for cand in surname_candidates:
            if re.search(r"\b" + re.escape(cand) + r"\s+et\s+al\b", norm_text):
                result.update({"present": True, "method": "citation", "details": {"candidate": cand}})
                matched = True
                break
            if re.search(r"\b" + re.escape(cand) + r"\s*[,(]\s*\d{4}\b", norm_text) or re.search(r"\b" + re.escape(cand) + r"\s*\(\s*\d{4}\s*\)", norm_text):
                result.update({"present": True, "method": "citation", "details": {"candidate": cand}})
                matched = True
                break
        if matched:
            results[author] = result
            continue

        # 2) Exact surname presence
        for cand in surname_candidates:
            if re.search(r"\b" + re.escape(cand) + r"\b", norm_text):
                result.update({"present": True, "method": "surname", "details": {"candidate": cand}})
                matched = True
                break
        if matched:
            results[author] = result
            continue

        # 3) Full name proximity: given token and surname appear within window
        if given_tokens:
            window = 5
            for cand in surname_candidates:
                cand_positions = [i for w, i in words_with_pos if w == cand]
                if not cand_positions:
                    continue
                found_prox = False
                for g in given_tokens:
                    g_positions = [i for w, i in words_with_pos if w == g]
                    if not g_positions:
                        continue
                    for p in cand_positions:
                        for q in g_positions:
                            if abs(p - q) <= window:
                                result.update({"present": True, "method": "fullname", "details": {"candidate": cand, "given": g, "positions": (q, p)}})
                                found_prox = True
                                break
                        if found_prox:
                            break
                    if found_prox:
                        break
                if found_prox:
                    matched = True
                    break
        if matched:
            results[author] = result
            continue

        # 4) Fuzzy surname matching (last resort)
        best_score = 0.0
        best_word = None
        best_cand = None
        for cand in surname_candidates:
            for w in words:
                if len(w) < 3 or len(cand) < 3:
                    continue
                score = difflib.SequenceMatcher(None, cand, w).ratio()
                if score > best_score:
                    best_score = score
                    best_word = w
                    best_cand = cand
        if best_score >= fuzzy_threshold:
            result.update({"present": True, "method": "fuzzy", "details": {"candidate": best_cand, "score": best_score, "matched_word": best_word}})
            matched = True

        results[author] = result

    return results


def get_matched_authors(authors: List[str], text: str, fuzzy_threshold: float = 0.92) -> List[str]:
    """Return list of original author strings that were matched in the text."""
    info = find_authors_in_text(authors, text, fuzzy_threshold=fuzzy_threshold)
    matched = [author for author, meta in info.items() if meta.get("present")]
    return matched


# -----------------------
# Example / quick tests
# -----------------------
# if __name__ == "__main__":
#     authors = ["D Bahdanau", "Christian Buck", "Kyunghyun Cho", "Sébastien Jean", "M.-T Luong"]
#     text = 'Neural Machine Translation (NMT) achieved state-of-the-art performances in large-scale translation tasks such as from English to French (Luong et al., 2015) and English to German (Jean et al., 2015). NMT is appealing since it requires minimal domain knowledge and is conceptually simple. The model by Luong et al. (2015) reads through all the source words until the end-ofsentence symbol <eos> is reached. It then starts emitting one target word at a time, as illustrated in Figure 1. NMT is often a large neural network that is trained in an end-to-end fashion and has the ability to generalize well to very long word sequences. This means the model does not have to explicitly store gigantic phrase tables and language models as in the case of standard MT; hence, NMT has a small memory footprint. Lastly, implementing NMT decoders is easy unlike the highly intricate decoders in standard MT (Koehn et al., 2003).\n\nIn parallel, the concept of "attention" has gained popularity recently in training neural networks, allowing models to learn alignments between different modalities, e.g., between image objects and agent actions in the dynamic control problem (Mnih et al., 2014), between speech frames and text in the speech recognition task (?), or between visual features of a picture and its text description in the image caption generation task (Xu et al., 2015). In the context of NMT, Bahdanau et al. (2015) has successfully applied such attentional mechanism to jointly translate and align words. To the best of our knowledge, there has not been any other work exploring the use of attention-based architectures for NMT.\n\nIn this work, we design, with simplicity and ef-fectiveness in mind, two novel types of attentionbased models: a global approach in which all source words are attended and a local one whereby only a subset of source words are considered at a time. The former approach resembles the model of (Bahdanau et al., 2015) but is simpler architecturally. The latter can be viewed as an interesting blend between the hard and soft attention models proposed in (Xu et al., 2015): it is computationally less expensive than the global model or the soft attention; at the same time, unlike the hard attention, the local attention is differentiable almost everywhere, making it easier to implement and train. 2 Besides, we also examine various alignment functions for our attention-based models.\n\nExperimentally, we demonstrate that both of our approaches are effective in the WMT translation tasks between English and German in both directions. Our attentional models yield a boost of up to 5.0 BLEU over non-attentional systems which already incorporate known techniques such as dropout. For English to German translation, we achieve new state-of-the-art (SOTA) results for both WMT\'14 and WMT\'15, outperforming previous SOTA systems, backed by NMT models and n-gram LM rerankers, by more than 1.0 BLEU. We conduct extensive analysis to evaluate our models in terms of learning, the ability to handle long sentences, choices of attentional architectures, alignment quality, and translation outputs.'
    
#     matched = get_matched_authors(authors, text)
#     print(matched)

    # # Additional tests for reversed order and full names:
    # text2 = "This follows work by Sebastien Jean and also by M.-T. Luong in subsequent studies."
    # res2 = find_authors_in_text(authors, text2)
    # print("\nSecond text:")
    # for a, info in res2.items():
    #     print(f"{a:20} -> present={info['present']:5} method={info['method']:8} matched={info['matched_string']} details={info['details']}")
