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
    args should be (query, db_path, top_k, candidate_limit, use_authors)
    """
    query, db_path, top_k, candidate_limit, use_authors = args
    results = search_paper(query, db_path, top_k=top_k, candidate_limit=candidate_limit, use_authors=use_authors)
    return results

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
        args_list.append((query, db_path, top_k, candidate_limit, use_authors))
    
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

# if __name__ == "__main__":
#     # Example paths
#     # ndjson_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson"
#     db_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_meta_4.db"

#     candidate_limit = 100   # tune down for speed, up for recall
#     top_k = 1
#     use_authors_flag = False  # title-only mode (fast). Set True to include authors.

#     # create / reuse a Searcher (opens one sqlite connection)

#     queries_1 = [
#         {'title': 'Neural Machine Translation by Jointly Learning to Align and Translate', 'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']},
#         {'title': 'Modeling outof-vocabulary words for robust speech recognition', 'authors_teiled': ['Issam Bazzi', 'James Glass']},
#         {'title': 'Compositional Morphology for Word Representations and Language Modelling', 'authors_teiled': ['Jan Botha', 'Phil Blunsom']},
#         {'title': 'Variable-Length Word Encodings for Neural Translation Models', 'authors_teiled': ['Rohan Chitnis', 'John Denero']},
#         {'title': 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation', 'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merrienboer', 'Caglar Gulcehre', 'Dzmitry Bahdanau', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio']},
#         {'title': 'Unsupervised Discovery of Morphemes', 'authors_teiled': ['Mathias Creutz', 'Krista Lagus']},
#         {'title': 'Integrating an Unsupervised Transliteration Model into Statistical Machine Translation', 'authors_teiled': ['Nadir Durrani', 'Hassan Sajjad', 'Hieu Hoang', 'Philipp Koehn']},
#         {'title': 'A Simple, Fast, and Effective Reparameterization of IBM Model 2', 'authors_teiled': ['Chris Dyer', 'Victor Chahuneau', 'Noah Smith']},
#         {'title': 'A New Algorithm for Data Compression', 'authors_teiled': ['Philip Gage']},
#         {'title': 'The Edinburgh/JHU Phrase-based Machine Translation Systems for WMT', 'authors_teiled': ['Barry Haddow', 'Matthias Huck', 'Alexandra Birch', 'Nikolay Bogoychev', 'Philipp Koehn']},
#         {'title': 'On Using Very Large Target Vocabulary for Neural Machine Translation', 'authors_teiled': ['Sébastien Jean', 'Kyunghyun Cho', 'Roland Memisevic', 'Yoshua Bengio']},
#         {'title': 'Recurrent Continuous Translation Models', 'authors_teiled': ['Nal Kalchbrenner', 'Phil Blunsom']},
#         {'title': 'Character-Aware Neural Language Models', 'authors_teiled': ['Yoon Kim', 'Yacine Jernite', 'David Sontag', 'Alexander Rush']},
#         {'title': 'Empirical Methods for Compound Splitting', 'authors_teiled': ['Philipp Koehn', 'Kevin Knight']},
#         {'title': 'Moses: Open Source Toolkit for Statistical Machine Translation', 'authors_teiled': ['Philipp Koehn', 'Hieu Hoang', 'Alexandra Birch', 'Chris Callison-Burch', 'Marcello Federico', 'Nicola Bertoldi', 'Brooke Cowan', 'Wade Shen', 'Christine Moran', 'Richard Zens', 'Chris Dyer', 'Ondřej Bojar', 'Alexandra Constantin', 'Evan Herbst']},
#         ]

#     queries = queries_1
#     # mapping = mapping_3


#     # see_act = list(mapping.values())
#     start_time = time.time()

#     # MULTI-THREADED VERSION
#     print("Using multi-threaded search...")
    
#     # Add act_val to each query for identification
#     # queries_with_act = []
#     # for query, act_val in zip(queries, mapping.keys()):
#     #     query_copy = query.copy()
#     #     query_copy['act_val'] = act_val
#     #     queries_with_act.append(query_copy)
    
#     # Perform parallel search
#     parallel_results = search_papers_parallel(
#         queries, 
#         db_path, 
#         top_k=1, 
#         use_authors=True,
#         max_workers=24  # Adjust based on your system capabilities
#     )
    
#     # Process results
#     res = []
#     for results in parallel_results:
#         if results:
#             for r in results:
#                 print(r)
#             res.append(True)
#         else:
#             print("[NOT FOUND]")
#             res.append(False)
#         print('-'*100)

#     end_time = time.time()

#     # true_count_list1 = see_act.count(True)
#     true_count_list2 = res.count(True)
#     print("#"*100)
#     # print(f"Actual[True] = {true_count_list1}, Found[True] = {true_count_list2}")
#     # print(see_act)
#     print(res)
#     print(f"Total time taken for the loop: {end_time - start_time} seconds")