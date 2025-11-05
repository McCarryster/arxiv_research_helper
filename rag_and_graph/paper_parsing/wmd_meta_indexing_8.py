import json
import sqlite3
import os
import re
import time
import unicodedata
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

# Try to import rapidfuzz for faster/better fuzzy matching
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    from difflib import SequenceMatcher
    _HAS_RAPIDFUZZ = False

# -------------------------
# Normalization utilities
# -------------------------
_RE_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_RE_WS = re.compile(r"\s+")

def normalize_text(s: Optional[str]) -> str:
    """
    Normalize text: NFKD normalize, remove diacritics, lowercase,
    remove punctuation, collapse whitespace.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    # remove accents
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s

def join_authors_from_record(rec: Dict[str, Any]) -> str:
    """
    Given a parsed JSON record, return a single string of authors
    (handles several possible shapes in arXiv metadata).
    """
    authors = []
    candidates = []
    if "metadata" in rec and isinstance(rec["metadata"], dict):
        md = rec["metadata"]
        if "authors" in md:
            candidates.append(md["authors"])
        if "author" in md:
            candidates.append(md["author"])
        if "forenames" in md or "keyname" in md:
            name_parts = []
            if "forenames" in md:
                name_parts.append(md.get("forenames", ""))
            if "keyname" in md:
                name_parts.append(md.get("keyname", ""))
            if any(name_parts):
                candidates.append([{"forenames": md.get("forenames",""), "keyname": md.get("keyname","")}])
    if "authors" in rec:
        candidates.append(rec["authors"])
    if "author" in rec:
        candidates.append(rec["author"])
    for cand in candidates:
        if isinstance(cand, dict):
            if "author" in cand and isinstance(cand["author"], list):
                for a in cand["author"]:
                    name = _extract_author_name(a)
                    if name:
                        authors.append(name)
            else:
                name = _extract_author_name(cand)
                if name:
                    authors.append(name)
        elif isinstance(cand, list):
            for a in cand:
                if isinstance(a, dict):
                    name = _extract_author_name(a)
                    if name:
                        authors.append(name)
                elif isinstance(a, str):
                    authors.append(a)
        elif isinstance(cand, str):
            authors.append(cand)
    if not authors:
        if "forenames" in rec or "keyname" in rec:
            fn = rec.get("forenames","")
            kn = rec.get("keyname","")
            if fn or kn:
                authors.append((fn + " " + kn).strip())
    seen = set()
    out = []
    for a in authors:
        na = a.strip()
        if na and na not in seen:
            seen.add(na)
            out.append(na)
    return ", ".join(out)

def _extract_author_name(a: dict) -> str:
    if not isinstance(a, dict):
        return ""
    fn = a.get("forenames") or a.get("forename") or ""
    kn = a.get("keyname") or a.get("lastname") or a.get("surname") or ""
    name = a.get("name") or a.get("fullname") or ""
    if name:
        return name
    parts = []
    if fn:
        parts.append(fn)
    if kn:
        parts.append(kn)
    return " ".join(parts).strip()

# -------------------------
# DB and index functions
# -------------------------
def _detect_fts_version(conn: sqlite3.Connection) -> str:
    """
    Return 'fts5' or 'fts4' depending on availability.
    """
    cur = conn.cursor()
    try:
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts_check USING fts5(x);")
        conn.commit()
        cur.execute("DROP TABLE IF EXISTS __fts_check;")
        conn.commit()
        return "fts5"
    except sqlite3.OperationalError:
        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts_check USING fts4(x);")
            conn.commit()
            cur.execute("DROP TABLE IF EXISTS __fts_check;")
            conn.commit()
            return "fts4"
        except sqlite3.OperationalError:
            raise RuntimeError("Neither FTS5 nor FTS4 available in this sqlite3 build.")

def build_index(ndjson_path: str, db_path: str, batch_size: int = 1000) -> None:
    """
    Build a sqlite DB with an FTS virtual table for title+authors+abstract.
    This function streams the ndjson file and inserts in batches.
    """
    if not os.path.exists(ndjson_path):
        raise FileNotFoundError(ndjson_path)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    fts_ver = _detect_fts_version(conn)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE papers (
        rowid INTEGER PRIMARY KEY,
        arxiv_id TEXT,
        title TEXT,
        title_norm TEXT,
        authors TEXT,
        authors_norm TEXT,
        abstract TEXT,
        categories TEXT,
        created TEXT
    );
    """)
    if fts_ver == "fts5":
        cur.execute("""
        CREATE VIRTUAL TABLE papers_fts USING fts5(title, authors, abstract, content='papers', content_rowid='rowid');
        """)
    else:  # fts4
        cur.execute("""
        CREATE VIRTUAL TABLE papers_fts USING fts4(title, authors, abstract);
        """)

    conn.commit()

    insert_sql = "INSERT INTO papers (arxiv_id, title, title_norm, authors, authors_norm, abstract, categories, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    fts_insert_sql = "INSERT INTO papers_fts(rowid, title, authors, abstract) VALUES (?, ?, ?, ?)"

    batch = []
    count = 0

    with open(ndjson_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            arxiv_id = None
            if isinstance(rec.get("id"), str):
                arxiv_id = rec.get("id")
            elif isinstance(rec.get("metadata"), dict) and isinstance(rec["metadata"].get("id"), str):
                arxiv_id = rec["metadata"]["id"]
            elif isinstance(rec.get("oai_header"), dict) and isinstance(rec["oai_header"].get("identifier"), str):
                arxiv_id = rec["oai_header"]["identifier"].split(":")[-1]
            else:
                arxiv_id = None

            title = ""
            if isinstance(rec.get("title"), str):
                title = rec.get("title")
            elif isinstance(rec.get("metadata"), dict) and isinstance(rec["metadata"].get("title"), str):
                title = rec["metadata"]["title"]

            abstract = ""
            if isinstance(rec.get("abstract"), str):
                abstract = rec.get("abstract")
            elif isinstance(rec.get("metadata"), dict) and isinstance(rec["metadata"].get("abstract"), str):
                abstract = rec["metadata"]["abstract"]

            authors = join_authors_from_record(rec)

            categories = ""
            if isinstance(rec.get("categories"), str):
                categories = rec.get("categories")
            elif isinstance(rec.get("metadata"), dict) and isinstance(rec["metadata"].get("categories"), str):
                categories = rec["metadata"].get("categories", "")

            created = ""
            if isinstance(rec.get("created"), str):
                created = rec.get("created")
            elif isinstance(rec.get("metadata"), dict) and isinstance(rec["metadata"].get("created"), str):
                created = rec["metadata"].get("created", "")

            title_norm = normalize_text(title)
            authors_norm = normalize_text(authors)
            batch.append((arxiv_id, title, title_norm, authors, authors_norm, abstract, categories, created))
            count += 1

            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                conn.commit()
                last_rowid = cur.execute("SELECT last_insert_rowid()").fetchone()[0]
                n = len(batch)
                start_rowid = last_rowid - n + 1
                for i, tup in enumerate(batch):
                    rid = start_rowid + i
                    cur.execute(fts_insert_sql, (rid, tup[1], tup[4], tup[5]))
                conn.commit()
                batch = []

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()
        last_rowid = cur.execute("SELECT last_insert_rowid()").fetchone()[0]
        n = len(batch)
        start_rowid = last_rowid - n + 1
        for i, tup in enumerate(batch):
            rid = start_rowid + i
            cur.execute(fts_insert_sql, (rid, tup[1], tup[4], tup[5]))
        conn.commit()

    cur.execute("CREATE INDEX idx_papers_arxiv_id ON papers(arxiv_id);")
    conn.commit()
    conn.close()
    print(f"Indexed {count} records into {db_path} (FTS={fts_ver}).")

# -------------------------
# Scoring / search
# -------------------------
def _fuzzy_score(a: str, b: str) -> float:
    """
    Return a similarity score in range [0, 100].
    Uses rapidfuzz if available; otherwise difflib ratio scaled to 0-100.
    """
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(a, b)
    else:
        return SequenceMatcher(None, a, b).ratio() * 100.0

def _escape_fts_token(token: str) -> str:
    """
    Basic safety: tokens are normalized (alnum + spaces) so this is minimal.
    If you expect weird tokens consider stronger escaping/quoting.
    """
    return token

def _build_fts_query(title: str, authors: List[str]) -> str:
    """
    Build an FTS MATCH query string.
    Title: keep AND between meaningful words.
    Authors: build a grouped expression per author that requires at least one token from each author:
        (a1 OR a2) AND (b1 OR b2) AND ...
    This avoids requiring every token from every author (which was too strict).
    """
    parts = []
    tnorm = normalize_text(title)
    if tnorm:
        words = [w for w in tnorm.split() if len(w) > 1]
        if words:
            # require title words (use AND)
            parts.append(" AND ".join(_escape_fts_token(w) for w in words))

    auth_subparts = []
    for a in authors:
        a_norm = normalize_text(a)
        tokens = [t for t in a_norm.split() if len(t) > 1]
        if not tokens:
            continue
        # create OR group for this author so at least one token of that author must match
        or_group = " OR ".join(_escape_fts_token(t) for t in tokens)
        if len(tokens) > 1:
            or_group = f"({or_group})"
        auth_subparts.append(or_group)

    if auth_subparts:
        # require each author group to match at least one of its tokens
        parts.append(" AND ".join(auth_subparts))

    if parts:
        return " AND ".join(parts)
    else:
        return ""

def _build_title_only_fts_query(title: str) -> str:
    """
    Build a title-only FTS MATCH string (faster) by AND-ing meaningful title tokens.
    """
    tnorm = normalize_text(title)
    if not tnorm:
        return ""
    words = [w for w in tnorm.split() if len(w) > 1]
    if not words:
        return ""
    return " AND ".join(_escape_fts_token(w) for w in words)

def _build_like_query_terms(title: str, authors: List[str]) -> Tuple[str, str]:
    # fallback for LIKE search (not used by default)
    return f"%{title}%", "%".join(authors) if authors else "%"

def search_paper(query: Dict[str, Any], db_path: str, top_k: int = 10, candidate_limit: int = 500, use_authors: bool = True) -> List[Dict[str, Any]]:
    """
    Search for a paper using query dict like:
       {'title': 'Neural Machine Translation by Jointly Learning to Align and Translate',
        'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']}

    New parameter:
      use_authors (bool): if False, use title-only FTS match and skip author matching (faster).
                         if True, use title+author MATCH (original behavior).

    Returns list of results with fields and a combined score.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    title = query.get("title", "") or ""
    authors_list = query.get("authors_teiled") or query.get("authors") or []
    authors_list = [str(a).strip() for a in authors_list if a and str(a).strip()]

    # If user requested title-only, ignore author list for building MATCH query and later scoring.
    if not use_authors:
        fts_query = _build_title_only_fts_query(title)
    else:
        fts_query = _build_fts_query(title, authors_list)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = []
    if fts_query:
        try:
            # use alias 'f' for FTS table and 'p' for papers
            sql = "SELECT p.rowid, p.arxiv_id, p.title, p.authors, p.abstract, p.categories, p.created FROM papers_fts f JOIN papers p ON p.rowid = f.rowid WHERE f MATCH ? LIMIT ?"
            cur.execute(sql, (fts_query, candidate_limit))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # fallback to LIKE if MATCH fails
            like_title, like_auth = _build_like_query_terms(title, authors_list)
            cur.execute("SELECT rowid, arxiv_id, title, authors, abstract, categories, created FROM papers WHERE title LIKE ? OR authors LIKE ? LIMIT ?", (like_title, like_auth, candidate_limit))
            rows = cur.fetchall()
    else:
        cur.execute("SELECT rowid, arxiv_id, title, authors, abstract, categories, created FROM papers ORDER BY rowid DESC LIMIT ?", (candidate_limit,))
        rows = cur.fetchall()

    title_norm_q = normalize_text(title)
    authors_norm_q = [normalize_text(a) for a in authors_list] if use_authors else []

    scored = []
    for r in rows:
        title_db = r["title"] or ""
        authors_db = r["authors"] or ""
        tn_db = normalize_text(title_db)
        an_db = normalize_text(authors_db)

        title_score = _fuzzy_score(title_norm_q, tn_db)
        if authors_norm_q:
            author_scores = []
            # compare each query author against whole authors string and individual authors in the DB
            chunks = [c.strip() for c in authors_db.split(",") if c.strip()]
            chunk_norms = [normalize_text(c) for c in chunks]
            for a_q in authors_norm_q:
                score_whole = _fuzzy_score(a_q, an_db)
                chunk_scores = [_fuzzy_score(a_q, cn) for cn in chunk_norms] if chunk_norms else []
                max_chunk = max([score_whole] + chunk_scores)
                author_scores.append(max_chunk)
            authors_score = sum(author_scores) / len(author_scores)
        else:
            authors_score = 0.0

        combined = (0.7 * title_score) + (0.3 * authors_score)

        scored.append((combined, title_score, authors_score, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for combined, tscore, ascore, r in scored[:top_k]:
        results.append({
            "arxiv_id": r["arxiv_id"],
            "title": r["title"],
            "authors": r["authors"],
            "abstract": r["abstract"],
            "categories": r["categories"],
            "created": r["created"],
            "score": float(combined),
            "title_score": float(tscore),
            "authors_score": float(ascore)
        })

    conn.close()
    return results

# -------------------------
# Multi-threaded search function
# -------------------------
def search_paper_wrapper(args):
    """
    Wrapper function for multi-threaded execution.
    args should be (query, db_path, top_k, candidate_limit, use_authors, act_val)
    """
    query, db_path, top_k, candidate_limit, use_authors, act_val = args
    results = search_paper(query, db_path, top_k=top_k, candidate_limit=candidate_limit, use_authors=use_authors)
    return results, act_val

def search_papers_parallel(queries: List[Dict[str, Any]], db_path: str, top_k: int = 10, 
                          candidate_limit: int = 500, use_authors: bool = True, 
                          max_workers: int = None) -> List[Tuple[List[Dict[str, Any]], str]]: # type: ignore
    """
    Search for multiple papers in parallel using multi-threading.
    
    Returns a list of tuples: (results, act_val) for each query
    """
    # Prepare arguments for each query
    args_list = []
    for query in queries:
        # Extract act_val from query if needed, or use empty string
        act_val = query.get('act_val', '')
        args_list.append((query, db_path, top_k, candidate_limit, use_authors, act_val))
    
    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(search_paper_wrapper, args) for args in args_list]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in parallel search: {e}")
                results.append(([], ''))
    
    return results

if __name__ == "__main__":
    # Example paths
    ndjson_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_metadata.ndjson"
    db_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_meta_4.db"

    candidate_limit = 100   # tune down for speed, up for recall
    top_k = 1
    use_authors_flag = False  # title-only mode (fast). Set True to include authors.

    # create / reuse a Searcher (opens one sqlite connection)

    queries_1 = [
        {'title': 'Neural Machine Translation by Jointly Learning to Align and Translate', 'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']},
        {'title': 'Modeling outof-vocabulary words for robust speech recognition', 'authors_teiled': ['Issam Bazzi', 'James Glass']},
        {'title': 'Compositional Morphology for Word Representations and Language Modelling', 'authors_teiled': ['Jan Botha', 'Phil Blunsom']},
        {'title': 'Variable-Length Word Encodings for Neural Translation Models', 'authors_teiled': ['Rohan Chitnis', 'John Denero']},
        {'title': 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation', 'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merrienboer', 'Caglar Gulcehre', 'Dzmitry Bahdanau', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio']},
        {'title': 'Unsupervised Discovery of Morphemes', 'authors_teiled': ['Mathias Creutz', 'Krista Lagus']},
        {'title': 'Integrating an Unsupervised Transliteration Model into Statistical Machine Translation', 'authors_teiled': ['Nadir Durrani', 'Hassan Sajjad', 'Hieu Hoang', 'Philipp Koehn']},
        {'title': 'A Simple, Fast, and Effective Reparameterization of IBM Model 2', 'authors_teiled': ['Chris Dyer', 'Victor Chahuneau', 'Noah Smith']},
        {'title': 'A New Algorithm for Data Compression', 'authors_teiled': ['Philip Gage']},
        {'title': 'The Edinburgh/JHU Phrase-based Machine Translation Systems for WMT', 'authors_teiled': ['Barry Haddow', 'Matthias Huck', 'Alexandra Birch', 'Nikolay Bogoychev', 'Philipp Koehn']},
        {'title': 'On Using Very Large Target Vocabulary for Neural Machine Translation', 'authors_teiled': ['Sébastien Jean', 'Kyunghyun Cho', 'Roland Memisevic', 'Yoshua Bengio']},
        {'title': 'Recurrent Continuous Translation Models', 'authors_teiled': ['Nal Kalchbrenner', 'Phil Blunsom']},
        {'title': 'Character-Aware Neural Language Models', 'authors_teiled': ['Yoon Kim', 'Yacine Jernite', 'David Sontag', 'Alexander Rush']},
        {'title': 'Empirical Methods for Compound Splitting', 'authors_teiled': ['Philipp Koehn', 'Kevin Knight']},
        {'title': 'Moses: Open Source Toolkit for Statistical Machine Translation', 'authors_teiled': ['Philipp Koehn', 'Hieu Hoang', 'Alexandra Birch', 'Chris Callison-Burch', 'Marcello Federico', 'Nicola Bertoldi', 'Brooke Cowan', 'Wade Shen', 'Christine Moran', 'Richard Zens', 'Chris Dyer', 'Ondřej Bojar', 'Alexandra Constantin', 'Evan Herbst']},
        ]
    mapping_1 = {
        "Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural Machine Translation by Jointly Learning to Align and Translate. In Proceedings of the International Conference on Learning Representations (ICLR).": True,
        "Issam Bazzi and James R. Glass. 2000. Modeling outof-vocabulary words for robust speech recognition. In Sixth International Conference on Spoken Language Processing, ICSLP 2000 / INTERSPEECH 2000, pages 401–404, Beijing, China": False,
        "Jan A. Botha and Phil Blunsom. 2014. Compositional Morphology for Word Representations and Language Modelling. In Proceedings of the 31st International Conference on Machine Learning (ICML), Beijing, China.": True,
        "Rohan Chitnis and John DeNero. 2015. VariableLength Word Encodings for Neural Translation Models. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).": False,
        "Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724–1734, Doha, Qatar. Association for Computational Linguistics.": True,
        "Mathias Creutz and Krista Lagus. 2002. Unsupervised Discovery of Morphemes. In Proceedings of the ACL-02 Workshop on Morphological and Phonological Learning, pages 21–30. Association for Computational Linguistics.": True,
        "Nadir Durrani, Hassan Sajjad, Hieu Hoang, and Philipp Koehn. 2014. Integrating an Unsupervised Transliteration Model into Statistical Machine Translation. In Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2014, pages 148–153, Gothenburg, Sweden.": False,
        "Chris Dyer, Victor Chahuneau, and Noah A. Smith. 2013. A Simple, Fast, and Effective Reparameterization of IBM Model 2. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 644–648, Atlanta, Georgia. Association for Computational Linguistics.": False,
        "Philip Gage. 1994. A New Algorithm for Data Compression. C Users J., 12(2):23–38, February.": False,
        "Barry Haddow, Matthias Huck, Alexandra Birch, Nikolay Bogoychev, and Philipp Koehn. 2015. The Edinburgh/JHU Phrase-based Machine Translation Systems for WMT 2015. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 126–133, Lisbon, Portugal. Association for Computational Linguistics": True, # Or false (hz)
        "Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. On Using Very Large Target Vocabulary for Neural Machine Translation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1–10, Beijing, China. Association for Computational Linguistics.": True,
        "Nal Kalchbrenner and Phil Blunsom. 2013. Recurrent Continuous Translation Models. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, Seattle. Association for Computational Linguistics.": False,
        "Yoon Kim, Yacine Jernite, David Sontag, and Alexander M. Rush. 2015. Character-Aware Neural Language Models. CoRR, abs/1508.06615.": True,
        "Philipp Koehn and Kevin Knight. 2003. Empirical Methods for Compound Splitting. In EACL ’03: Proceedings of the Tenth Conference on European Chapter of the Association for Computational Linguistics, pages 187–193, Budapest, Hungary. Association for Computational Linguistics": True,
        "Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran, Richard Zens, Chris Dyer, Ondˇrej Bojar, Alexandra Constantin, and Evan Herbst. 2007. Moses: Open Source Toolkit for Statistical Machine Translation. In Proceedings of the ACL-2007 Demo and Poster Sessions, pages 177–180, Prague, Czech Republic. Association for Computational Linguistics.": False,
    }


    queries_2 = [
        {'title': 'Neural machine translation by jointly learning to align and translate', 'authors_teiled': ['D Bahdanau', 'K Cho', 'Y Bengio']},
        {'title': 'N-gram counts and language models from the common crawl', 'authors_teiled': ['Christian Buck', 'Kenneth Heafield', 'Bas Van Ooyen']},
        {'title': 'Learning phrase representations using RNN encoder-decoder for statistical machine translation', 'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merrienboer', 'Caglar Gulcehre', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio']},
        {'title': 'Measuring word alignment quality for statistical machine translation', 'authors_teiled': ['Alexander Fraser', 'Daniel Marcu']},
        {'title': 'DRAW: A recurrent neural network for image generation', 'authors_teiled': ['Karol Gregor', 'Ivo Danihelka', 'Alex Graves', 'Danilo Jimenez Rezende', 'Daan Wierstra']},
        {'title': 'On using very large target vocabulary for neural machine translation', 'authors_teiled': ['Sébastien Jean', 'Kyunghyun Cho', 'Roland Memisevic', 'Yoshua Bengio']},
        {'title': 'Recurrent continuous translation models', 'authors_teiled': ['N Kalchbrenner', 'P Blunsom']},
        {'title': 'Statistical phrase-based translation', 'authors_teiled': ['Philipp Koehn', 'Franz', 'Josef Och', 'Daniel Marcu']},
        {'title': 'Alignment by agreement', 'authors_teiled': ['P Liang', 'B Taskar', 'D Klein']},
        {'title': 'Addressing the rare word problem in neural machine translation', 'authors_teiled': ['M.-T Luong', 'I Sutskever', 'Q Le', 'O Vinyals', 'W Zaremba']},
        {'title': 'Recurrent models of visual attention', 'authors_teiled': ['Volodymyr Mnih', 'Nicolas Heess', 'Alex Graves', 'Koray Kavukcuoglu']},
        {'title': 'Bleu: a method for automatic evaluation of machine translation', 'authors_teiled': ['Kishore Papineni', 'Salim Roukos', 'Todd Ward', 'Wei Jing', 'Zhu']},
        {'title': 'Sequence to sequence learning with neural networks', 'authors_teiled': ['I Sutskever', 'O Vinyals', 'Q Le']},
        {'title': 'Show, attend and tell: Neural image caption generation with visual attention', 'authors_teiled': ['Kelvin Xu', 'Jimmy Ba', 'Ryan Kiros', 'Kyunghyun Cho', 'Aaron Courville', 'Ruslan Salakhutdinov', 'Richard Zemel', 'Yoshua Bengio']},
        {'title': 'Recurrent neural network regularization', 'authors_teiled': ['Wojciech Zaremba', 'Ilya Sutskever', 'Oriol Vinyals']},
        ]
    mapping_2 = {
        "D. Bahdanau, K. Cho, and Y. Bengio. 2015. Neural machine translation by jointly learning to align and translate. In ICLR.": True,
        "Christian Buck, Kenneth Heafield, and Bas van Ooyen. 2014. N-gram counts and language models from the common crawl. In LREC.": False,
        "Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In EMNLP.": True,
        "Alexander Fraser and Daniel Marcu. 2007. Measuring word alignment quality for statistical machine translation. Computational Linguistics, 33(3):293–303.": False,
        "Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, and Daan Wierstra. 2015. DRAW: A recurrent neural network for image generation. In ICML.": True,
        "S´ebastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. On using very large target vocabulary for neural machine translation. In ACL.": True,
        "N. Kalchbrenner and P. Blunsom. 2013. Recurrent continuous translation models. In EMNLP.": False,
        "Philipp Koehn, Franz Josef Och, and Daniel Marcu. 2003. Statistical phrase-based translation. In NAACL.": False,
        "P. Liang, B. Taskar, and D. Klein. 2006. Alignment by agreement. In NAACL.": False,
        "M.-T. Luong, I. Sutskever, Q. V. Le, O. Vinyals, and W. Zaremba. 2015. Addressing the rare word problem in neural machine translation. In ACL.": True,
        "Volodymyr Mnih, Nicolas Heess, Alex Graves, and Koray Kavukcuoglu. 2014. Recurrent models of visual attention. In NIPS.": True,
        "Kishore Papineni, Salim Roukos, Todd Ward, and Wei jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In ACL.": False,
        "I. Sutskever, O. Vinyals, and Q. V. Le. 2014. Sequence to sequence learning with neural networks. In NIPS.": True,
        "Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron C. Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Bengio. 2015. Show, attend and tell: Neural image caption generation with visual attention. In ICML.": True,
        "Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. 2015. Recurrent neural network regularization. In ICLR.": True,
    }


    queries_3 = [
        {'title': 'Imagenet classification with deep convolutional neural networks', 'authors_teiled': ['Alex Krizhevsky', 'Ilya Sutskever', 'Geoffrey Hinton']},
        {'title': 'Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups', 'authors_teiled': ['Geoffrey Hinton', 'Li Deng', 'Dong Yu', 'George Dahl', 'Abdel-Rahman Mohamed', 'Navdeep Jaitly', 'Andrew Senior', 'Vincent Vanhoucke', 'Patrick Nguyen', 'Tara Sainath']},
        {'title': 'Large scale distributed deep networks', 'authors_teiled': ['Jeffrey Dean', 'Greg Corrado', 'Rajat Monga', 'Kai Chen', 'Matthieu Devin', 'Mark Mao', 'Andrew Senior', 'Paul Tucker', 'Ke Yang', 'Quoc V Le']},
        {'title': 'Batch normalization: Accelerating deep network training by reducing internal covariate shift', 'authors_teiled': ['Sergey Ioffe', 'Christian Szegedy']},
        {'title': 'Sequence to sequence learning with neural networks', 'authors_teiled': ['Ilya Sutskever', 'Oriol Vinyals', 'Quoc V Le']},
        {'title': 'Batch normalized recurrent neural networks', 'authors_teiled': ['César Laurent', 'Gabriel Pereyra', 'Philémon Brakel', 'Ying Zhang', 'Yoshua Bengio']},
        {'title': 'Deep speech 2: End-to-end speech recognition in english and mandarin', 'authors_teiled': ['Dario Amodei', 'Rishita Anubhai', 'Eric Battenberg', 'Carl Case', 'Jared Casper', 'Bryan Catanzaro', 'Jingdong Chen', 'Mike Chrzanowski', 'Adam Coates', 'Greg Diamos']},
        {'title': 'Recurrent batch normalization', 'authors_teiled': ['Tim Cooijmans', 'Nicolas Ballas', 'César Laurent', 'Aaron Courville']},
        {'title': 'Weight normalization: A simple reparameterization to accelerate training of deep neural networks', 'authors_teiled': ['Tim Salimans', 'P Diederik', 'Kingma']},
        {'title': 'Path-sgd: Path-normalized optimization in deep neural networks', 'authors_teiled': []},
        {'title': 'Natural gradient works efficiently in learning', 'authors_teiled': ['Shun-Ichi Amari']},
        {'title': 'Order-embeddings of images and language', 'authors_teiled': ['Ivan Vendrov', 'Ryan Kiros', 'Sanja Fidler', 'Raquel Urtasun']},
        {'title': 'A python framework for fast computation of mathematical expressions', 'authors_teiled': ['The Theano', 'Development Team', 'Rami Al-Rfou', 'Guillaume Alain', 'Amjad Almahairi', 'Christof Angermueller', 'Dzmitry Bahdanau', 'Nicolas Ballas', 'Frédéric Bastien', 'Justin Bayer', 'Anatoly Belikov']},
        {'title': 'Microsoft coco: Common objects in context', 'authors_teiled': ['Tsung-Yi Lin', 'Michael Maire', 'Serge Belongie', 'James Hays', 'Pietro Perona', 'Deva Ramanan', 'Piotr Dollár', 'C Lawrence', 'Zitnick']},
        {'title': 'Learning phrase representations using rnn encoder-decoder for statistical machine translation', 'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merriënboer', 'Caglar Gulcehre', 'Dzmitry Bahdanau', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio']},
    ]
    mapping_3 = {
        "Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.": False,
        "Geoffrey Hinton, Li Deng, Dong Yu, George E Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE, 2012.": False,
        "Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Andrew Senior, Paul Tucker, Ke Yang, Quoc V Le, et al. Large scale distributed deep networks. In NIPS, 2012.": False,
        "Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. ICML, 2015.": True,
        "Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pages 3104–3112, 2014.": True,
        "Cesar Laurent, Gabriel Pereyra, Phil ´ emon Brakel, Ying Zhang, and Yoshua Bengio. Batch normalized recurrent neural networks. arXiv preprint arXiv:1510.01378, 2015.": True,
        "Dario Amodei, Rishita Anubhai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Jingdong Chen, Mike Chrzanowski, Adam Coates, Greg Diamos, et al. Deep speech 2: End-to-end speech recognition in english and mandarin. arXiv preprint arXiv:1512.02595, 2015.": True,
        "Tim Cooijmans, Nicolas Ballas, Cesar Laurent, and Aaron Courville. Recurrent batch normalization. ´ arXiv preprint arXiv:1603.09025, 2016.": True,
        "Tim Salimans and Diederik P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. arXiv preprint arXiv:1602.07868, 2016.": True,
        "Behnam Neyshabur, Ruslan R Salakhutdinov, and Nati Srebro. Path-sgd: Path-normalized optimization in deep neural networks. In Advances in Neural Information Processing Systems, pages 2413–2421, 2015.": True,
        "Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural computation, 1998": False,
        "Ivan Vendrov, Ryan Kiros, Sanja Fidler, and Raquel Urtasun. Order-embeddings of images and language. ICLR, 2016.": True,
        "The Theano Development Team, Rami Al-Rfou, Guillaume Alain, Amjad Almahairi, Christof Angermueller, Dzmitry Bahdanau, Nicolas Ballas, Frederic Bastien, Justin Bayer, Anatoly Belikov, et al. Theano: A python ´ framework for fast computation of mathematical expressions. arXiv preprint arXiv:1605.02688, 2016.": True,
        "Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and ´ C Lawrence Zitnick. Microsoft coco: Common objects in context. ECCV, 2014.": True,
        "Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. EMNLP, 2014.": True,
    }

    queries = queries_2
    mapping = mapping_2


    see_act = list(mapping.values())
    start_time = time.time()

    # MULTI-THREADED VERSION
    print("Using multi-threaded search...")
    
    # Add act_val to each query for identification
    queries_with_act = []
    for query, act_val in zip(queries, mapping.keys()):
        query_copy = query.copy()
        query_copy['act_val'] = act_val
        queries_with_act.append(query_copy)
    
    # Perform parallel search
    parallel_results = search_papers_parallel(
        queries_with_act, 
        db_path, 
        top_k=1, 
        use_authors=False,
        max_workers=12  # Adjust based on your system capabilities
    )
    
    # Process results
    res = []
    for results, act_val in parallel_results:
        if results:
            for r in results:
                print(r)
            res.append(True)
        else:
            print("[NOT FOUND]", act_val)
            res.append(False)
        print('-'*100)

    end_time = time.time()

    true_count_list1 = see_act.count(True)
    true_count_list2 = res.count(True)
    print("#"*100)
    print(f"Actual[True] = {true_count_list1}, Found[True] = {true_count_list2}")
    print(see_act)
    print(res)
    print(f"Total time taken for the loop: {end_time - start_time} seconds")