import json
import os
import re
import time
import unicodedata
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

# DB driver
try:
    import psycopg2
    import psycopg2.extras
except Exception as e:
    raise ImportError("psycopg2 is required. Install with: pip install psycopg2-binary") from e

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
# Postgres helpers / schema
# -------------------------

def _get_conn(pg_dsn: str):
    """
    Open a new psycopg2 connection. Caller should close it.
    pg_dsn can be a libpq-style DSN or connection string.
    Example: "dbname=mydb user=me password=secret host=localhost port=5432"
    """
    conn = psycopg2.connect(pg_dsn)
    return conn

def _detect_or_create_extensions(conn: psycopg2.extensions.connection) -> None:
    """
    Ensure pg_trgm extension exists (for trigram similarity / indexes).
    If the DB user lacks permissions, the CREATE EXTENSION will fail but that's okay:
    we continue without trigram optimizations.
    """
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        conn.commit()
    except Exception:
        conn.rollback()

def create_schema(conn: psycopg2.extensions.connection) -> None:
    """
    Create tables and indexes needed for search. Safe to call multiple times.
    - papers table with title_norm & authors_norm stored (pre-normalized)
    - search_tsv tsvector column combining title/authors/abstract for FTS
    - GIN index on search_tsv
    - GIN trigram indexes on title_norm and authors_norm (if pg_trgm available)
    """
    cur = conn.cursor()
    _detect_or_create_extensions(conn)

    # Main table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        id BIGSERIAL PRIMARY KEY,
        arxiv_id TEXT,
        title TEXT,
        title_norm TEXT,
        authors TEXT,
        authors_norm TEXT,
        abstract TEXT,
        categories TEXT,
        created TEXT,
        search_tsv tsvector -- combined vector for FTS
    );
    """)
    conn.commit()

    # Update trigger function to maintain search_tsv
    cur.execute("""
    CREATE OR REPLACE FUNCTION papers_search_tsv_trigger() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      NEW.search_tsv :=
         setweight(to_tsvector('simple', coalesce(NEW.title, '')), 'A')
         || setweight(to_tsvector('simple', coalesce(NEW.authors, '')), 'B')
         || setweight(to_tsvector('simple', coalesce(NEW.abstract, '')), 'C');
      RETURN NEW;
    END;
    $$;
    """)
    conn.commit()

    # Create trigger
    cur.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'papers_search_tsv_trigger' AND tgrelid = 'papers'::regclass
      ) THEN
        CREATE TRIGGER papers_search_tsv_trigger
        BEFORE INSERT OR UPDATE ON papers
        FOR EACH ROW EXECUTE FUNCTION papers_search_tsv_trigger();
      END IF;
    END;
    $$;
    """)
    conn.commit()

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_search_tsv ON papers USING gin (search_tsv);")
    conn.commit()

    # Trigram indexes (if pg_trgm available, creation will succeed; otherwise ignore errors)
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_title_trgm ON papers USING gin (title_norm gin_trgm_ops);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_authors_trgm ON papers USING gin (authors_norm gin_trgm_ops);")
        conn.commit()
    except Exception:
        conn.rollback()

    # Fast lookup on arxiv_id
    cur.execute("CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id);")
    conn.commit()

# -------------------------
# Ingest / build index
# -------------------------
def build_index(ndjson_path: str, pg_dsn: str, batch_size: int = 1000) -> None:
    """
    Build (or rebuild) the papers table in PostgreSQL from an ndjson file.
    - ndjson_path: path to ndjson file (one JSON record per line)
    - pg_dsn: libpq dsn string for psycopg2.connect
    """
    if not os.path.exists(ndjson_path):
        raise FileNotFoundError(ndjson_path)

    conn = _get_conn(pg_dsn)
    try:
        cur = conn.cursor()
        # Create schema (tables, triggers, indexes)
        create_schema(conn)

        # Optionally truncate existing data
        cur.execute("TRUNCATE TABLE papers;")
        conn.commit()

        insert_sql = """
        INSERT INTO papers
          (arxiv_id, title, title_norm, authors, authors_norm, abstract, categories, created)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """

        batch = []
        count = 0
        start_time = time.time()
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
                    psycopg2.extras.execute_batch(cur, insert_sql, batch)
                    conn.commit()
                    batch = []

        # final batch
        if batch:
            psycopg2.extras.execute_batch(cur, insert_sql, batch)
            conn.commit()

        # After insert, ensure search_tsv is populated for any pre-existing rows (trigger handles it on insert,
        # but to be safe for any older rows we run an update)
        cur.execute("UPDATE papers SET search_tsv = setweight(to_tsvector('simple', coalesce(title,'')),'A') || setweight(to_tsvector('simple', coalesce(authors,'')),'B') || setweight(to_tsvector('simple', coalesce(abstract,'')),'C') WHERE search_tsv IS NULL;")
        conn.commit()

        elapsed = time.time() - start_time
        print(f"Indexed {count} records into Postgres (elapsed {elapsed:.1f}s).")
    finally:
        conn.close()

# -------------------------
# Scoring / search utilities
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
        return SequenceMatcher(None, a, b).ratio() * 100.0  # type: ignore[name-defined]

def _escape_ts_token(tok: str) -> str:
    """
    Minimal cleaning for tokens to be used in to_tsquery('simple', ...)
    Replace single quotes, AND/OR chars. Keep only word chars.
    """
    if not tok:
        return ""
    tok = re.sub(r"[^\w]", " ", tok)
    tok = re.sub(r"\s+", " ", tok).strip()
    return tok

def _build_pg_tsquery(title: str, authors: List[str], use_authors: bool) -> str:
    """
    Build a tsquery string for to_tsquery('simple', ...).
    - Title tokens are ANDed.
    - Authors produce an OR group (author phrases and individual tokens).
    Example: "neural & machine & (bahdanau | bahdanau bahdanau | bahdanau cho)"
    """
    parts = []
    tnorm = normalize_text(title)
    if tnorm:
        title_tokens = [w for w in tnorm.split() if len(w) > 1]
        if title_tokens:
            # to_tsquery expects tokens separated by & for AND
            tparts = []
            for w in title_tokens:
                w2 = _escape_ts_token(w)
                if w2:
                    tparts.append(w2)
            if tparts:
                parts.append(" & ".join(tparts))

    if use_authors and authors:
        author_pieces = []
        seen = set()
        for a in authors:
            a_norm = normalize_text(a)
            if not a_norm:
                continue
            # full author phrase
            phrase = _escape_ts_token(a_norm)
            if phrase and phrase not in seen:
                seen.add(phrase)
                author_pieces.append(phrase)
            # individual tokens
            tokens = [t for t in a_norm.split() if len(t) > 1]
            for t in tokens:
                t_esc = _escape_ts_token(t)
                if t_esc and t_esc not in seen:
                    seen.add(t_esc)
                    author_pieces.append(t_esc)
        if author_pieces:
            parts.append("(" + " | ".join(author_pieces) + ")")

    if parts:
        return " & ".join(parts)
    else:
        return ""

def _fetch_candidates_fallback_pg(cur: psycopg2.extensions.cursor, title_norm_q: str, authors_norm_q_list: List[str], candidate_limit: int) -> List[Dict[str, Any]]:
    """
    Fallback candidate fetch:
      1) Try ILIKE on normalized columns (title_norm, authors_norm)
      2) If nothing, fetch most recent candidate_limit rows
    Returns list of dict rows.
    """
    rows = []
    tpattern = f"%{title_norm_q}%" if title_norm_q else "%"
    authors_join = " ".join(authors_norm_q_list) if authors_norm_q_list else ""
    apattern = f"%{authors_join}%" if authors_join else "%"

    try:
        cur.execute("""
            SELECT id, arxiv_id, title, authors, abstract, categories, created
            FROM papers
            WHERE title_norm ILIKE %s OR authors_norm ILIKE %s
            LIMIT %s
        """, (tpattern, apattern, candidate_limit))
        rows = [dict(r) for r in cur.fetchall()]
        if rows:
            return rows
    except Exception:
        # fall through
        pass

    # final fallback: most recent rows
    cur.execute("""
      SELECT id, arxiv_id, title, authors, abstract, categories, created
      FROM papers
      ORDER BY id DESC
      LIMIT %s
    """, (candidate_limit,))
    rows = [dict(r) for r in cur.fetchall()]
    return rows

# -------------------------
# Search implementation
# -------------------------
def search_paper(query: Dict[str, Any], pg_dsn: str, top_k: int = 10, candidate_limit: int = 500, use_authors: bool = True) -> List[Dict[str, Any]]:
    """
    Search for a paper using query dict like:
       {'title': 'Neural Machine Translation by Jointly Learning to Align and Translate',
        'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']}
    pg_dsn: connection string for psycopg2.connect
    Returns: list of result dicts (same keys as your original search_paper)
    """
    title = query.get("title", "") or ""
    # accept both keys authors_teiled (your input) or authors
    authors_list = query.get("authors_teiled") or query.get("authors") or []
    authors_list = [str(a).strip() for a in authors_list if a and str(a).strip()]

    conn = _get_conn(pg_dsn)
    try:
        # conn.row_factory = None  # not used
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Build tsquery
        tsquery_str = _build_pg_tsquery(title, authors_list, use_authors)
        rows = []
        used_fts = False

        if tsquery_str:
            try:
                # Use to_tsquery('simple', ...) for token/phrase matching; rank by ts_rank
                cur.execute("""
                    SELECT id, arxiv_id, title, authors, abstract, categories, created,
                           ts_rank_cd(search_tsv, to_tsquery('simple', %s)) AS rank
                    FROM papers
                    WHERE search_tsv @@ to_tsquery('simple', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, (tsquery_str, tsquery_str, candidate_limit))
                rows = [dict(r) for r in cur.fetchall()]
                used_fts = True
            except Exception:
                # if to_tsquery fails or something else, fall back
                conn.rollback()
                rows = []

        if not rows:
            # Fallback candidates using normalized LIKE or recent rows
            title_norm_q = normalize_text(title)
            authors_norm_q_list = [normalize_text(a) for a in authors_list] if use_authors else []
            rows = _fetch_candidates_fallback_pg(cur, title_norm_q, authors_norm_q_list, candidate_limit)

        # Prepare normalized query strings for scoring (python-side fuzzy)
        title_norm_q = normalize_text(title)
        authors_norm_q = [normalize_text(a) for a in authors_list] if use_authors else []

        scored: List[Tuple[float, float, float, Dict[str, Any]]] = []
        for r in rows:
            title_db = r.get("title") or ""
            authors_db = r.get("authors") or ""
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
                "arxiv_id": r.get("arxiv_id"),
                "title": r.get("title"),
                "authors": r.get("authors"),
                "abstract": r.get("abstract"),
                "categories": r.get("categories"),
                "created": r.get("created"),
                "score": float(combined),
                "title_score": float(tscore),
                "authors_score": float(ascore)
            })

        return results
    finally:
        conn.close()

# -------------------------
# Multi-threaded search
# -------------------------
def search_paper_wrapper(args):
    """
    Wrapper for multi-threaded execution.
    args should be (query, pg_dsn, top_k, candidate_limit, use_authors)
    """
    query, pg_dsn, top_k, candidate_limit, use_authors = args
    return search_paper(query, pg_dsn, top_k=top_k, candidate_limit=candidate_limit, use_authors=use_authors)

def search_papers_parallel(queries: List[Dict[str, Any]], pg_dsn: str, top_k: int = 1,
                          candidate_limit: int = 500, use_authors: bool = True,
                          max_workers: int = None) -> List[List[Dict[str, Any]]]: # type: ignore
    """
    Search for multiple papers in parallel. Returns a list of result-lists.
    Each worker opens its own DB connection (safe for psycopg2).
    """
    args_list = []
    for query in queries:
        args_list.append((query, pg_dsn, top_k, candidate_limit, use_authors))

    results: List[List[Dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(search_paper_wrapper, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                r = future.result()
                results.append(r)
            except Exception as e:
                # best-effort: return empty result for that query
                print(f"Error in parallel search: {e}")
                results.append([])

    return results


if __name__ == "__main__":
    # Example paths
    ndjson_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson"

    candidate_limit = 100   # tune down for speed, up for recall
    top_k = 1
    use_authors_flag = False  # title-only mode (fast). Set True to include authors.
    # build_index(db_path, )          
    # build_index(ndjson_path, "dbname=arxiv_meta_db user=postgres password=@Q_Fa;ml$f!@94r host=localhost port=5432")
    # create / reuse a Searcher (opens one sqlite connection)

    # queries_1 = [
    #     {'title': 'Exploring the limits of language modeling', 'authors_teiled': ['Rafal Jozefowicz', 'Oriol Vinyals', 'Mike Schuster', 'Noam Shazeer', 'Yonghui Wu']},
    #     {'title': 'Outrageously large neural networks: The sparsely-gated mixture-of-experts layer', 'authors_teiled': ['Noam Shazeer', 'Azalia Mirhoseini', 'Krzysztof Maziarz', 'Andy Davis', 'Quoc Le', 'Geoffrey Hinton', 'Jeff Dean']},
    #     {'title': 'A decomposable attention model', 'authors_teiled': ['Ankur Parikh', 'Oscar Täckström', 'Dipanjan Das', 'Jakob Uszkoreit']},
    #     {'title': 'Factorization tricks for LSTM networks', 'authors_teiled': ['Oleksii Kuchaiev', 'Boris Ginsburg']},
    #     {'title': 'Empirical evaluation of gated recurrent neural networks on sequence modeling', 'authors_teiled': ['Junyoung Chung', 'Çaglar Gülçehre', 'Kyunghyun Cho', 'Yoshua Bengio']},
    #     {'title': 'Google’s neural machine translation system: Bridging the gap between human and machine translation', 'authors_teiled': ['Yonghui Wu', 'Mike Schuster', 'Zhifeng Chen', 'V Quoc', 'Mohammad Le', 'Wolfgang Norouzi', 'Maxim Macherey', 'Yuan Krikun', 'Qin Cao', 'Klaus Gao', 'Macherey']},
    #     {'title': 'Structured attention networks', 'authors_teiled': ['Yoon Kim', 'Carl Denton', 'Luong Hoang', 'Alexander M Rush']},
    #     {'title': 'Learning phrase representations using rnn encoder-decoder for statistical machine translation', 'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merrienboer', 'Caglar Gulcehre', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio']},
    #     {'title': 'Neural machine translation by jointly learning to align and translate', 'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']},
    #     {'title': 'Sequence to sequence learning with neural networks', 'authors_teiled': ['Ilya Sutskever', 'Oriol Vinyals', 'Quoc Vv Le']},
    #     {'title': 'Effective approaches to attention- based neural machine translation', 'authors_teiled': ['Minh-Thang Luong', 'Hieu Pham', 'Christopher D Manning']},
    #     {'title': 'Long short-term memory', 'authors_teiled': ['Sepp Hochreiter', 'Jürgen Schmidhuber']},
    #     ]

    queries_2 = [
        {'title': 'A Scalable Hierarchical Distributed Language Model', 'authors_teiled': ['A Mnih', 'G Hinton']},
        {'title': 'Gradient Flow in Recurrent Nets: the Diﬃculty of Learning Long-term Dependencies', 'authors_teiled': ['S Hochreiter', 'Y Bengio', 'P Frasconi', 'J Schmidhuber', 'S C Kremer', 'J F Kolen']},
        {'title': 'Lecture 6.5 - rmsprop: Divide the gradient by a running average of its recent magnitude', 'authors_teiled': ['T Tieleman', 'G Hinton']},
        {'title': 'A machine learning perspective on predictive coding with paq', 'authors_teiled': ['B Knoll', 'N De Freitas']},
        {'title': 'Data compression using adaptive cod- ing and partial string matching', 'authors_teiled': ['J G Cleary', 'Ian', 'I H Witten']},
        {'title': 'Better generative models for sequential data problems: Bidi- rectional recurrent mixture density networks', 'authors_teiled': ['M Schuster']},
        {'title': 'Subword language modeling with neural networks', 'authors_teiled': ['T Mikolov', 'I Sutskever', 'A Deoras', 'H Le', 'S Kombrink', 'J Cernocky']},
        {'title': 'Framewise phoneme classiﬁcation with bidi- rectional LSTM and other neural network architectures', 'authors_teiled': ['A Graves', 'J Schmidhuber']},
        {'title': 'The tagged LOB corpus user’s manual', 'authors_teiled': ['S Johansson', 'R Atwell', 'R Garside', 'G Leech']},
        {'title': 'The recurrent temporal restricted boltzmann machine', 'authors_teiled': ['I Sutskever', 'G E Hinton', 'G W Taylor']},
        {'title': 'The Minimum Description Length Principle (Adaptive Computation and Machine Learning)', 'authors_teiled': ['P D Gr¨unwald']},
        {'title': 'Oﬄine handwriting recognition with multi- dimensional recurrent neural networks', 'authors_teiled': ['A Graves', 'J Schmidhuber']},
        {'title': 'Generating text with recurrent neural networks', 'authors_teiled': ['I Sutskever', 'J Martens', 'G Hinton']},
        {'title': 'A ﬁrst look at music composition using lstm recurrent neural networks', 'authors_teiled': ['D Eck', 'J Schmidhuber']},
        {'title': 'Practical variational inference for neural networks', 'authors_teiled': ['A Graves']},
        {'title': 'Statistical Language Models based on Neural Networks', 'authors_teiled': ['T Mikolov']},
        {'title': 'Building a large annotated corpus of english: The penn treebank', 'authors_teiled': ['M P Marcus', 'B Santorini', 'M A Marcinkiewicz']},
        {'title': 'A Practical Guide to Training Restricted Boltzmann Machines', 'authors_teiled': ['G Hinton']},
        {'title': 'Low- rank matrix factorization for deep neural network training with high- dimensional output targets', 'authors_teiled': ['T N Sainath', 'A Mohamed', 'B Kingsbury', 'B Ramabhadran']},
        {'title': 'Long Short-Term Memory', 'authors_teiled': ['S Hochreiter', 'J Schmidhuber']},
        {'title': 'Speech recognition with deep recurrent neural networks', 'authors_teiled': ['A Graves', 'A Mohamed', 'G Hinton']},
        {'title': 'Learning precise timing with LSTM recurrent networks', 'authors_teiled': ['F Gers', 'N Schraudolph', 'J Schmidhuber']},
        {'title': 'IAM-OnDB - an on-line English sentence database acquired from handwritten text on a whiteboard', 'authors_teiled': ['M Liwicki', 'H Bunke']},
        {'title': 'Gradient-based learning algorithms for recur- rent networks and their computational complexity', 'authors_teiled': ['R Williams', 'D Zipser']},
        {'title': 'The Human Knowledge Compression Contest', 'authors_teiled': ['M Hutter']},
        {'title': 'Factored conditional restricted boltzmann machines for modeling motion style', 'authors_teiled': ['G W Taylor', 'G E Hinton']},
        {'title': 'Learning long-term dependencies with gradient descent is diﬃcult', 'authors_teiled': ['Y Bengio', 'P Simard', 'P Frasconi']},
        {'title': 'Modeling tempo- ral dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription', 'authors_teiled': ['N Boulanger-Lewandowski', 'Y Bengio', 'P Vincent']},
        {'title': 'Neural Networks for Pattern Recognition', 'authors_teiled': ['C Bishop']},
        {'title': 'An analysis of noise in recurrent neural networks: convergence and generalization', 'authors_teiled': ['K.-C Jim', 'C Giles', 'B Horne']},
        {'title': 'Sequence transduction with recurrent neural networks', 'authors_teiled': ['A Graves']},
        {'title': 'A fast and simple algorithm for training neural probabilistic language models', 'authors_teiled': ['A Mnih', 'Y W Teh']},
        {'title': 'Mixture density networks', 'authors_teiled': ['C Bishop']},
    ]

    queries = queries_2
    # mapping = mapping_3


    # see_act = list(mapping.values())
    start_time = time.time()

    # MULTI-THREADED VERSION
    print("Using multi-threaded search...")
    
    # Add act_val to each query for identification
    # queries_with_act = []
    # for query, act_val in zip(queries, mapping.keys()):
    #     query_copy = query.copy()
    #     query_copy['act_val'] = act_val
    #     queries_with_act.append(query_copy)
    
    # Perform parallel search
    parallel_results = search_papers_parallel(queries, "dbname=arxiv_meta_db user=postgres password=@Q_Fa;ml$f!@94r host=localhost port=5432", top_k=1, max_workers=12)
    
    # Process results
    for i, res in enumerate(parallel_results):
        if res:
            print(i+1, res[0]['arxiv_id'], res[0]['title'], res[0]['authors']) # type: ignore
        else:
            print(i, "[NOT FOUND]")
        print("-"*100)

    end_time = time.time()

    print(f"Total time taken for the loop: {end_time - start_time} seconds")