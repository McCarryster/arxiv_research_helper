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
    # remove accents / combining diacritics
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
                    # Insert original title and original authors into FTS table (let sqlite tokenize)
                    cur.execute(fts_insert_sql, (rid, tup[1], tup[3], tup[5]))
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
            cur.execute(fts_insert_sql, (rid, tup[1], tup[3], tup[5]))
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
    Minimal escaping for FTS MATCH tokens.
    Surround token with double quotes if it contains spaces (phrase).
    """
    if not token:
        return token
    if " " in token:
        token = token.replace('"', '""')
        return f'"{token}"'
    return token

def _build_fts_query(title: str, authors: List[str]) -> str:
    """
    Build an FTS MATCH query string.

    Title: keep AND between meaningful words.

    Authors: build a permissive author expression so at least one author token/phrase
             must match (OR across author tokens/phrases).
    """
    parts = []
    tnorm = normalize_text(title)
    if tnorm:
        words = [w for w in tnorm.split() if len(w) > 1]
        if words:
            parts.append(" AND ".join(_escape_fts_token(w) for w in words))

    author_pieces = []
    for a in authors:
        a_norm = normalize_text(a)
        if not a_norm:
            continue
        # full author phrase
        author_pieces.append(_escape_fts_token(a_norm))
        # individual tokens
        tokens = [t for t in a_norm.split() if len(t) > 1]
        for t in tokens:
            author_pieces.append(_escape_fts_token(t))

    if author_pieces:
        seen = set()
        uniq = []
        for x in author_pieces:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        parts.append("(" + " OR ".join(uniq) + ")")

    if parts:
        return " AND ".join(parts)
    else:
        return ""

def _build_title_only_fts_query(title: str) -> str:
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

def _fetch_candidates_fallback(cur: sqlite3.Cursor, title_norm_q: str, authors_norm_q_list: List[str], candidate_limit: int) -> List[sqlite3.Row]:
    """
    Fallback candidate fetch when FTS MATCH yields nothing:
      1) Try normalized columns with LIKE (title_norm, authors_norm).
      2) If that yields nothing, fetch the most recent candidate_limit rows for fuzzy rescoring.
    """
    rows = []
    # Try title_norm LIKE first
    tpattern = f"%{title_norm_q}%" if title_norm_q else "%"
    # Combine author norms into one pattern (catch any author token)
    authors_join = " ".join(authors_norm_q_list) if authors_norm_q_list else ""
    apattern = f"%{authors_join}%" if authors_join else "%"

    try:
        cur.execute("SELECT rowid, arxiv_id, title, authors, abstract, categories, created FROM papers WHERE title_norm LIKE ? OR authors_norm LIKE ? LIMIT ?", (tpattern, apattern, candidate_limit))
        rows = cur.fetchall()
        if rows:
            return rows
    except Exception:
        # best-effort; fall through to broader fetch
        rows = []

    # If no rows, fetch most recent candidate_limit rows for fuzzy scoring
    cur.execute("SELECT rowid, arxiv_id, title, authors, abstract, categories, created FROM papers ORDER BY rowid DESC LIMIT ?", (candidate_limit,))
    rows = cur.fetchall()
    return rows

def search_paper(query: Dict[str, Any], db_path: str, top_k: int = 10, candidate_limit: int = 500, use_authors: bool = True) -> List[Dict[str, Any]]:
    """
    Search for a paper using query dict like:
       {'title': 'Neural Machine Translation by Jointly Learning to Align and Translate',
        'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']}
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    title = query.get("title", "") or ""
    # accept both keys authors_teiled (your input) or authors
    authors_list = query.get("authors_teiled") or query.get("authors") or []
    authors_list = [str(a).strip() for a in authors_list if a and str(a).strip()]

    if not use_authors:
        fts_query = _build_title_only_fts_query(title)
    else:
        fts_query = _build_fts_query(title, authors_list)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = []
    used_fts = False
    if fts_query:
        try:
            sql = "SELECT p.rowid, p.arxiv_id, p.title, p.authors, p.abstract, p.categories, p.created FROM papers_fts f JOIN papers p ON p.rowid = f.rowid WHERE f MATCH ? LIMIT ?"
            cur.execute(sql, (fts_query, candidate_limit))
            rows = cur.fetchall()
            used_fts = True
        except sqlite3.OperationalError:
            rows = []

    # If FTS returned nothing, try normalized LIKE fallback and finally recent rows fuzzy scan
    if not rows:
        title_norm_q = normalize_text(title)
        authors_norm_q_list = [normalize_text(a) for a in authors_list] if use_authors else []
        rows = _fetch_candidates_fallback(cur, title_norm_q, authors_norm_q_list, candidate_limit)

    # Prepare normalized query strings for scoring
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
    Wrapper for multi-threaded execution.
    args should be (query, db_path, top_k, candidate_limit, use_authors)
    """
    query, db_path, top_k, candidate_limit, use_authors = args
    return search_paper(query, db_path, top_k=top_k, candidate_limit=candidate_limit, use_authors=use_authors)

def search_papers_parallel(queries: List[Dict[str, Any]], db_path: str, top_k: int = 1, 
                          candidate_limit: int = 500, use_authors: bool = True, 
                          max_workers: int = None) -> List[List[Dict[str, Any]]]: # type: ignore
    """
    Search for multiple papers in parallel. Returns a list of result-lists.
    """
    args_list = []
    for query in queries:
        args_list.append((query, db_path, top_k, candidate_limit, use_authors))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(search_paper_wrapper, args) for args in args_list]
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in parallel search: {e}")
                results.append([])
    
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
#         {'title': 'Exploring the limits of language modeling', 'authors_teiled': ['Rafal Jozefowicz', 'Oriol Vinyals', 'Mike Schuster', 'Noam Shazeer', 'Yonghui Wu']},
#         {'title': 'Outrageously large neural networks: The sparsely-gated mixture-of-experts layer', 'authors_teiled': ['Noam Shazeer', 'Azalia Mirhoseini', 'Krzysztof Maziarz', 'Andy Davis', 'Quoc Le', 'Geoffrey Hinton', 'Jeff Dean']},
#         {'title': 'A decomposable attention model', 'authors_teiled': ['Ankur Parikh', 'Oscar Täckström', 'Dipanjan Das', 'Jakob Uszkoreit']},
#         {'title': 'Factorization tricks for LSTM networks', 'authors_teiled': ['Oleksii Kuchaiev', 'Boris Ginsburg']},
#         {'title': 'Empirical evaluation of gated recurrent neural networks on sequence modeling', 'authors_teiled': ['Junyoung Chung', 'Çaglar Gülçehre', 'Kyunghyun Cho', 'Yoshua Bengio']},
#         {'title': 'Google’s neural machine translation system: Bridging the gap between human and machine translation', 'authors_teiled': ['Yonghui Wu', 'Mike Schuster', 'Zhifeng Chen', 'V Quoc', 'Mohammad Le', 'Wolfgang Norouzi', 'Maxim Macherey', 'Yuan Krikun', 'Qin Cao', 'Klaus Gao', 'Macherey']},
#         {'title': 'Structured attention networks', 'authors_teiled': ['Yoon Kim', 'Carl Denton', 'Luong Hoang', 'Alexander M Rush']},
#         {'title': 'Learning phrase representations using rnn encoder-decoder for statistical machine translation', 'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merrienboer', 'Caglar Gulcehre', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio']},
#         {'title': 'Neural machine translation by jointly learning to align and translate', 'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']},
#         {'title': 'Sequence to sequence learning with neural networks', 'authors_teiled': ['Ilya Sutskever', 'Oriol Vinyals', 'Quoc Vv Le']},
#         {'title': 'Effective approaches to attention- based neural machine translation', 'authors_teiled': ['Minh-Thang Luong', 'Hieu Pham', 'Christopher D Manning']},
#         {'title': 'Long short-term memory', 'authors_teiled': ['Sepp Hochreiter', 'Jürgen Schmidhuber']},
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
#     for i, res in enumerate(parallel_results):
#         if res:
#             print(i+1, res[0]['arxiv_id'], res[0]['title'], res[0]['authors']) # type: ignore
#         else:
#             print(i, "[NOT FOUND]")
#         print("-"*100)

#     end_time = time.time()

#     print(f"Total time taken for the loop: {end_time - start_time} seconds")