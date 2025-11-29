import json
import re
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import requests

import psycopg2
import psycopg2.extras
from psycopg2 import pool
from psycopg2.extras import execute_values
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm.auto import tqdm
from config import *

class ArxivMetaSearchDB:
    def __init__(self, PG: dict, pool_minconn: int = 1, pool_maxconn: int = 10):
        """
        PG: dict with psycopg2.connect parameters, e.g. {"host":..., "dbname":..., "user":..., "password":...}
        pool_minconn/pool_maxconn: controls the threaded connection pool size.
        """
        # single connection for one-off / administrative tasks (schema, index creation, inserts)
        self.conn = psycopg2.connect(**PG)

        # Thread-safe connection pool for multi-threaded searches.
        # minconn should be at least 1, maxconn should be >= number of worker threads you plan to use.
        self.pool = pool.ThreadedConnectionPool(pool_minconn, pool_maxconn, **PG)

    # -------------------------
    # Pooled cursor helper
    # -------------------------
    @contextmanager
    def pooled_cursor(self, cursor_factory=None):
        """
        Context manager that yields a cursor borrowed from the threaded pool and
        returns the connection after use.

        Usage:
            with self.pooled_cursor() as cur:
                cur.execute(...)
                rows = cur.fetchall()
        """
        conn = None
        try:
            conn = self.pool.getconn()
            cur = conn.cursor(cursor_factory=cursor_factory) if cursor_factory else conn.cursor()
            try:
                yield cur
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
        finally:
            if conn is not None:
                # Return connection back to pool
                try:
                    self.pool.putconn(conn)
                except Exception:
                    pass

    # -------------------------
    # Normalization utilities
    # -------------------------
    def normalize_text(self, title: str) -> str:
        if title is None:
            return ""
        # Step 1: Unicode normalization to remove accents
        normalized = unicodedata.normalize("NFKD", str(title))
        normalized = normalized.encode("ASCII", "ignore").decode("utf-8")
        # Step 2: Lowercase
        normalized = normalized.lower()
        # Step 3: Remove unwanted characters (keep letters, numbers, and spaces)
        normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
        # Step 4: Replace multiple spaces with one and strip
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def format_authors(self, data: Any) -> List[str]:
        """
        Convert possible author structures into a list of "keyname forenames" strings.
        Accepts data as dict, list of dicts, or already list of strings.
        """
        if data is None:
            return []
        if isinstance(data, str):
            return [data]
        if isinstance(data, dict):
            data = [data]

        result: List[str] = []
        for item in data:
            if isinstance(item, str):
                result.append(item)
                continue
            if not isinstance(item, dict):
                continue
            keyname = item.get("keyname", "") or ""
            forenames = item.get("forenames", "") or ""
            keyname = str(keyname).strip()
            forenames = str(forenames).strip()
            if forenames:
                result.append(f"{keyname} {forenames}")
            else:
                result.append(keyname)
        return result

    def normalize_author_list(self, authors: Optional[List[str]]) -> List[str]:
        if authors is None:
            return []
        return [self.normalize_text(a) for a in authors]


    def prepare_queries(self, queries: List[dict]) -> List[Dict[str, Any]]:
        """
        Build queries_to_run from incoming queries.
        Each output dict contains:
            - original_citation: str or None (propagated from input)
            - glued_normalized_title: str (spaces removed)
            - normalized_authors: List[str] or None
        """
        queries_to_run: List[Dict[str, Any]] = []
        for q in queries:
            title_raw = q.get("title", "")
            original_citation = q.get("original_citation") or q.get("raw", "")
            authors_raw = q.get("authors_teiled", [])
            glued_normalized_title = self.normalize_text(title_raw)
            normalized_authors = self.normalize_author_list(authors_raw)
            queries_to_run.append({
                    "original_citation": original_citation,
                    "glued_normalized_title": glued_normalized_title.replace(" ", ""),
                    "normalized_authors": normalized_authors,
                })
        return queries_to_run

    # -------------------------
    # Schema + index creation
    # -------------------------
    def _create_schema(self) -> None:
        create_arxiv_paper_query = """
        CREATE TABLE IF NOT EXISTS PAPERS (
            id SERIAL PRIMARY KEY,
            arxiv_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            glued_normalized_title TEXT NOT NULL,
            normalized_authors TEXT[] NOT NULL,
            title_and_authors TEXT NOT NULL,
            created_date DATE NOT NULL,
            oai_header JSONB DEFAULT '{}'::jsonb,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(create_arxiv_paper_query)
        self.conn.commit()

    def _create_index(self) -> None:
        with self.conn.cursor() as cur:
            # B-tree index for fast equality lookups
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_papers_glued_normalized_title
                ON PAPERS (glued_normalized_title);
                """
            )

            # Try to create trigram extension + GIN trigram index for fast ILIKE / similarity.
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_papers_glued_normalized_title_trgm
                    ON PAPERS USING gin (glued_normalized_title gin_trgm_ops);
                    """
                )
            except Exception as e:
                # Non-fatal: print and continue
                print("Warning: could not create trigram index or extension:", e)
        self.conn.commit()
        print("Index creation attempted (B-tree + optional trigram).")

    def check_index(self) -> Dict[str, Any]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT extname FROM pg_extension WHERE extname IN ('pg_trgm', 'unaccent');"
            )
            exts = cur.fetchall()
            cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'papers';")
            indexes = cur.fetchall()
        info = {"extensions_installed": exts, "indexes": indexes}
        print(info)
        return info

    # -------------------------
    # Insert NDJSON
    # -------------------------
    def _insert_ndjson(self, ndjson_path: str) -> Dict[str, Any]:
        """
        Insert NDJSON to PAPERS table. Uses the single administrative connection (self.conn).
        Shows progress using tqdm.
        Returns a small summary dict.
        """
        inserted = 0
        skipped = 0
        errors = 0

        def get_num_lines(file_path: str) -> int:
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)

        total_lines = get_num_lines(ndjson_path)
        with open(ndjson_path, "r", encoding="utf-8") as fh:
            with self.conn.cursor() as cur:
                for line in tqdm(fh, total=total_lines, desc="Inserting NDJSON", unit="lines"):
                    line = line.strip()
                    if not line:
                        skipped += 1
                        continue
                    try:
                        paper = json.loads(line)
                    except json.JSONDecodeError:
                        errors += 1
                        continue

                    metadata = paper.get("metadata", {}) or {}
                    arxiv_id = metadata.get("id")
                    title = metadata.get("title", "") or ""
                    normalized_title = self.normalize_text(title)
                    glued_normalized_title = normalized_title.replace(" ", "")

                    authors_raw = metadata.get("authors", {}).get("author", []) if isinstance(metadata.get("authors", {}), dict) else metadata.get("authors", [])
                    authors = sorted(self.format_authors(authors_raw))
                    normalized_authors = self.normalize_author_list(authors)

                    title_and_authors = title + " | " + ", ".join(normalized_authors)

                    created_date_str = metadata.get("created", "")
                    created_date = None
                    if created_date_str:
                        try:
                            created_date = datetime.strptime(created_date_str, "%Y-%m-%d").date()
                        except Exception:
                            # fallback: try to parse only date portion
                            try:
                                created_date = datetime.fromisoformat(created_date_str.split("T")[0]).date()
                            except Exception:
                                created_date = None

                    oai_header = json.dumps(paper.get("oai_header", {}) or {})
                    metadata_json = json.dumps(metadata or {})

                    insert_query = """
                    INSERT INTO PAPERS (arxiv_id, title, normalized_title, glued_normalized_title, normalized_authors, title_and_authors, created_date, oai_header, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (arxiv_id) DO NOTHING;
                    """
                    try:
                        cur.execute(
                            insert_query,
                            (
                                arxiv_id,
                                title,
                                normalized_title,
                                glued_normalized_title,
                                normalized_authors,
                                title_and_authors,
                                created_date or datetime.utcnow().date(),
                                oai_header,
                                metadata_json,
                            ),
                        )
                        # Note: committing after the loop is OK, but if file is huge you may prefer batching commits.
                        inserted += 1
                    except Exception:
                        errors += 1
                        # do not stop the whole process; continue inserting other lines
                        try:
                            self.conn.rollback()
                        except Exception:
                            pass

        # commit all inserts
        try:
            self.conn.commit()
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass

        return {"inserted": inserted, "skipped_blank": skipped, "errors": errors}

    def _add_citation_counts_openalex(self, show_progress: bool = True) -> None:
        """
        Create citation_count column if missing and populate it using multithreaded
        OpenAlex lookups. Improvements for speed:
        - concurrent HTTP fetches (ThreadPoolExecutor)
        - session with retries/backoff
        - collect all results in memory and perform a single bulk update using a
            temporary table + INSERT ... (execute_values) + single UPDATE statement
            (much faster than one UPDATE per row)
        - number of workers is derived from the pool max connections (fallbacks provided)

        Parameters
        ----------
        show_progress:
            Whether to show a tqdm progress bar during network fetches.
        """


        # 1) Ensure column exists (single admin connection)
        with self.conn.cursor() as cur:
            cur.execute("ALTER TABLE PAPERS ADD COLUMN IF NOT EXISTS citation_count INTEGER;")
        self.conn.commit()

        # 2) Read distinct arXiv IDs using pooled cursor
        with self.pooled_cursor() as cur:
            cur.execute("SELECT DISTINCT arxiv_id FROM PAPERS WHERE arxiv_id IS NOT NULL;")
            rows = cur.fetchall()
        arxiv_ids: List[str] = [r[0] for r in rows] if rows else []

        if not arxiv_ids:
            return

        # 3) Prepare a requests Session with retries/backoff
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retries, pool_maxsize=50)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        # polite user-agent for API usage
        session.headers.update({"User-Agent": "arxiv-meta-search-db/1.0 (+https://example.org)"})

        BASE_URL = "https://api.openalex.org/works"
        per_page_params = {"per-page": 1, "select": "cited_by_count,ids"}

        # 4) decide number of workers (attempt to use pool maxconn, fallback to sensible defaults)
        pool_maxconn = getattr(self.pool, "maxconn", None) or getattr(self.pool, "_maxconn", None)
        try:
            max_workers = int(pool_maxconn) if pool_maxconn else min(32, max(4, (len(arxiv_ids) // 2) or 4))
        except Exception:
            max_workers = 8
        # keep an upper cap to avoid too many concurrent HTTP connections
        max_workers = max(4, min(max_workers, 64))

        def _fetch_for_arxiv(arxiv_id: str) -> Tuple[str, Optional[int]]:
            """
            Returns tuple (arxiv_id, cited_by_count_or_None)
            """
            aid = (arxiv_id or "").strip()
            if not aid:
                return arxiv_id, None

            # Attempt direct filter by ids.arxiv
            try:
                r = session.get(BASE_URL, params={"filter": f"ids.arxiv:{aid}", **per_page_params}, timeout=20)
                r.raise_for_status()
                payload = r.json()
                results = payload.get("results") or []
                if results:
                    v = results[0].get("cited_by_count")
                    return arxiv_id, int(v) if v is not None else None
            except Exception:
                # let fallback try; do not raise
                pass

            # Fallback: search by arXiv identifier forms
            for q in (aid, f"arXiv:{aid}"):
                try:
                    r2 = session.get(BASE_URL, params={"search": q, **per_page_params}, timeout=20)
                    r2.raise_for_status()
                    payload2 = r2.json()
                    results2 = payload2.get("results") or []
                    if not results2:
                        continue
                    candidate = results2[0]
                    ids_map = candidate.get("ids") or {}
                    # verify the arXiv id is present in any ids value
                    matched = False
                    for vv in ids_map.values():
                        try:
                            if isinstance(vv, str) and aid in vv:
                                matched = True
                                break
                        except Exception:
                            continue
                    if matched:
                        v = candidate.get("cited_by_count")
                        return arxiv_id, int(v) if v is not None else None
                except Exception:
                    continue

            return arxiv_id, None

        # 5) Run concurrent fetches and collect results
        results: List[Tuple[str, Optional[int]]] = []
        iterator = tqdm(arxiv_ids, desc="OpenAlex lookups", unit="id") if show_progress else None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_fetch_for_arxiv, aid): aid for aid in arxiv_ids}
            for fut in as_completed(futures):
                try:
                    aid, cnt = fut.result()
                    results.append((aid, cnt))
                except Exception:
                    # If a worker raises unexpectedly, treat as not found
                    failed_aid = futures.get(fut)
                    results.append((failed_aid, None)) # type: ignore
                if iterator is not None:
                    iterator.update(1)

        if iterator is not None:
            iterator.close()

        if not results:
            return

        # 6) Bulk update DB in one shot using a temporary table and execute_values
        # Use the administrative single connection self.conn for the bulk operation.
        rows_to_update = [(aid, cnt) for (aid, cnt) in results if aid is not None]
        if not rows_to_update:
            return

        with self.conn.cursor() as cur:
            # create a temporary table that will be dropped at transaction end
            cur.execute(
                "CREATE TEMP TABLE tmp_openalex_updates (arxiv_id TEXT PRIMARY KEY, citation_count INTEGER) ON COMMIT DROP;"
            )
            # bulk insert into temp table
            execute_values(
                cur,
                "INSERT INTO tmp_openalex_updates (arxiv_id, citation_count) VALUES %s",
                rows_to_update,
                template="(%s, %s)",
                page_size=1000,
            )
            # single UPDATE statement joining to the temp table
            cur.execute(
                """
                UPDATE PAPERS p
                SET citation_count = t.citation_count
                FROM tmp_openalex_updates t
                WHERE p.arxiv_id = t.arxiv_id;
                """
            )
        # commit the bulk update
        try:
            self.conn.commit()
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass

        return

    # -------------------------
    # build_db orchestration
    # -------------------------
    def build_db(self, ndjson_path: str, dry_run: bool = True) -> Dict[str, Any]:
        self._create_schema()
        insert_summary = self._insert_ndjson(ndjson_path)
        dup_summary = self.remove_duplicate_papers(dry_run=dry_run)
        self.vacuum_tables()
        self._create_index()
        idx_summary = self.check_index()
        self._add_citation_counts_openalex()
        return {"insert": insert_summary, "duplicates": dup_summary, "indexes": idx_summary}

    # -------------------------
    # Duplicate removal
    # -------------------------
    def remove_duplicate_papers(self, dry_run: bool = False) -> Dict[str, Any]:
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT 1 FROM (
                            SELECT
                                ROW_NUMBER() OVER (
                                    PARTITION BY title_and_authors
                                    ORDER BY created_date DESC, id DESC
                                ) AS rn
                            FROM PAPERS
                        ) t
                        WHERE t.rn > 1
                    ) s;
                    """
                )
                dup_count = cur.fetchone()[0]  # type: ignore

                if dup_count == 0:
                    return {"duplicates_found": 0, "deleted": 0, "deleted_ids": []}

                if dry_run:
                    return {"duplicates_found": dup_count, "deleted": 0, "deleted_ids": []}

                cur.execute(
                    """
                    WITH duplicates AS (
                        SELECT id FROM (
                            SELECT
                                id,
                                ROW_NUMBER() OVER (
                                    PARTITION BY title_and_authors
                                    ORDER BY created_date DESC, id DESC
                                ) AS rn
                            FROM PAPERS
                        ) t
                        WHERE t.rn > 1
                    )
                    DELETE FROM PAPERS p
                    USING duplicates d
                    WHERE p.id = d.id
                    RETURNING p.id;
                    """
                )
                deleted_rows = cur.fetchall()
                deleted_ids = [row[0] for row in deleted_rows]

            self.conn.commit()

            return {"duplicates_found": dup_count, "deleted": len(deleted_ids), "deleted_ids": deleted_ids}

        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            raise

    # -------------------------
    # VACUUM helper
    # -------------------------
    def vacuum_tables(self) -> None:
        try:
            self.conn.commit()
        except Exception:
            pass
        prev_autocommit = getattr(self.conn, "autocommit", False)
        self.conn.autocommit = True
        try:
            with self.conn.cursor() as cur:
                cur.execute("VACUUM ANALYZE PAPERS;")
        finally:
            self.conn.autocommit = prev_autocommit
        print("vacuum completed on PAPERS")

    # -------------------------
    # Main search function (thread-safe)
    # -------------------------
    def search_paper(self, original_citation: str, glued_normalized_title: str, query_authors: Optional[List[str]]=None) -> Any:
        """
        Thread-safe search: borrows a connection from the pool for each call.

        Returns:
            dict mapping columns -> values for a selected match, or False if not found/acceptable.
        """
        if not glued_normalized_title:
            return {original_citation: False}

        N_TITLE_CANDIDATES = 50
        import re
        from typing import List, Any, Optional, Set

        def _author_tokens(name: str) -> List[str]:
            """
            Break a name into lowercased tokens, removing empty tokens and common punctuation.
            Example: "Howard, Andrew G." -> ["howard", "andrew", "g"]
            """
            if not name:
                return []
            s = str(name).strip().lower()
            if not s:
                return []
            # remove surrounding braces/quotes
            if s.startswith("{") and s.endswith("}"):
                s = s[1:-1].strip()
            s = s.strip('"').strip("'")
            # replace punctuation (except hyphen) with spaces, keep hyphenated parts together
            s = re.sub(r"[^\w\-\s]", " ", s)
            tokens = [t for t in re.split(r"\s+", s) if t]
            # strip trailing dots from initials like "g." -> "g"
            tokens = [t.rstrip(".") for t in tokens]
            return tokens

        def _normalize_db_authors_field(db_val: Any) -> List[str]:
            """
            Preserve your original DB authors parsing behavior (handles arrays, postgres-style quoted lists, etc.) — returns list of author strings.
            """
            if db_val is None:
                return []
            if isinstance(db_val, (list, tuple)):
                return [str(x) for x in db_val]
            if isinstance(db_val, str):
                v = db_val.strip()
                if v.startswith("{") and v.endswith("}"):
                    v = v[1:-1]
                parts = []
                cur = []
                in_quote = False
                i = 0
                while i < len(v):
                    ch = v[i]
                    if ch == '"' and (i == 0 or v[i - 1] != "\\"):
                        in_quote = not in_quote
                    elif ch == "," and not in_quote:
                        part = "".join(cur).strip()
                        if part.startswith('"') and part.endswith('"'):
                            part = part[1:-1]
                        parts.append(part)
                        cur = []
                    else:
                        cur.append(ch)
                    i += 1
                if cur:
                    part = "".join(cur).strip()
                    if part.startswith('"') and part.endswith('"'):
                        part = part[1:-1]
                    if part:
                        parts.append(part)
                return [p for p in parts if p]
            return [str(db_val)]

        def _tokens_match(q_tokens: Set[str], db_tokens: Set[str]) -> bool:
            """
            Return True if at least one token from q_tokens matches db_tokens.
            Matching rules:
            - exact token equality
            - single-letter token in query matches first letter of a db token (initial match)
            - single-letter token in db matches first letter of a query token
            """
            if not q_tokens or not db_tokens:
                return False
            # direct intersection
            if q_tokens & db_tokens:
                return True
            # initial-based matching
            for qt in q_tokens:
                if len(qt) == 1:
                    # match query initial to db token startswith
                    for dt in db_tokens:
                        if dt and dt[0] == qt:
                            return True
            for dt in db_tokens:
                if len(dt) == 1:
                    for qt in q_tokens:
                        if qt and qt[0] == dt:
                            return True
            return False

        def _best_candidate_by_authors(rows: List[Any], cols: List[str], query_authors_list: Optional[List[str]]):
            # If no authors provided -> return top row if present
            if not query_authors_list:
                if not rows:
                    return None
                return dict(zip(cols, rows[0]))

            # Build token sets for each query author (preserve per-author sets)
            query_tokens_per_author: List[Set[str]] = []
            for a in query_authors_list:
                toks = set(_author_tokens(a))
                if toks:
                    query_tokens_per_author.append(toks)

            # If no usable tokens -> fallback to top row
            if not query_tokens_per_author:
                if not rows:
                    return None
                return dict(zip(cols, rows[0]))

            best_row = None
            best_score = 0
            best_created_date = None

            try:
                idx_auth = cols.index("normalized_authors")
            except ValueError:
                idx_auth = None
            try:
                idx_date = cols.index("created_date")
            except ValueError:
                idx_date = None

            for r in rows:
                db_auth_list = []
                if idx_auth is not None:
                    db_auth_field = r[idx_auth]
                    db_auth_list = _normalize_db_authors_field(db_auth_field)

                # Precompute token sets for all DB authors in this row
                db_tokens_list = [set(_author_tokens(db_a)) for db_a in db_auth_list]

                # For scoring, count how many distinct query authors are matched (1 per query author)
                matched_query_authors = 0
                for q_tokens in query_tokens_per_author:
                    matched = False
                    for db_tokens in db_tokens_list:
                        if _tokens_match(q_tokens, db_tokens):
                            matched = True
                            break
                    if matched:
                        matched_query_authors += 1

                score = matched_query_authors

                if score > best_score:
                    best_score = score
                    best_row = r
                    if idx_date is not None:
                        best_created_date = r[idx_date]
                elif score == best_score and score > 0:
                    # tie-breaker: prefer later created_date if available
                    if idx_date is not None and best_created_date is not None:
                        try:
                            if r[idx_date] and r[idx_date] > best_created_date:
                                best_row = r
                                best_created_date = r[idx_date]
                        except Exception:
                            pass

            if best_score > 0 and best_row is not None:
                return dict(zip(cols, best_row))
            return None

        # Use pooled cursor for thread safety (each thread uses its own connection from pool)
        with self.pooled_cursor() as cur:
            # 1) exact equality lookup
            cur.execute(
                """
                SELECT
                    id,
                    arxiv_id,
                    title,
                    citation_count,
                    normalized_title,
                    normalized_authors,
                    title_and_authors,
                    metadata,
                    created_date
                FROM PAPERS
                WHERE glued_normalized_title = %s
                ORDER BY created_date DESC
                LIMIT %s;
                """,
                (glued_normalized_title, N_TITLE_CANDIDATES),
            )
            exact_rows = cur.fetchall()
            cols = [d[0] for d in cur.description]

            if exact_rows:
                best = _best_candidate_by_authors(exact_rows, cols, query_authors)
                if best:
                    # return best
                    return {original_citation: best}
                return {original_citation: False}

            # 2) trigram similarity fallback (if available)
            try:
                cur.execute(
                    """
                    SELECT
                        id,
                        arxiv_id,
                        title,
                        citation_count,
                        normalized_title,
                        normalized_authors,
                        title_and_authors,
                        metadata,
                        created_date
                    FROM PAPERS
                    WHERE glued_normalized_title % %s
                    ORDER BY similarity(glued_normalized_title, %s) DESC
                    LIMIT %s;
                    """,
                    (glued_normalized_title, glued_normalized_title, N_TITLE_CANDIDATES),
                )
                trig_rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
            except Exception:
                # trigram not available or fails -> treat as no title matches
                return {original_citation: False}

            if not trig_rows:
                return {original_citation: False}

            best = _best_candidate_by_authors(trig_rows, cols, query_authors)
            if best:
                # return best
                return {original_citation: best}
            return {original_citation: False}

    def search_paper_by_arxiv_id(self, arxiv_id: str, original_citation: Optional[str]=None) -> Any:
        """
        Thread-safe search: borrows a connection from the pool for each call.

        Args:
            arxiv_id: The arXiv ID to search for.
            original_citation: Optional key for wrapping the result dictionary.

        Returns:
            dict mapping columns -> values for the selected match, or False if not found.
            If original_citation is provided, returns {original_citation: result} for consistency.
        """
        if not arxiv_id:
            return {original_citation: False} if original_citation else False

        # Use pooled cursor for thread safety (each thread uses its own connection from pool)
        with self.pooled_cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM PAPERS
                WHERE arxiv_id = %s
                LIMIT 1;
                """,
                (arxiv_id,)
            )
            row = cur.fetchone()
            if row is None:
                return {original_citation: False} if original_citation else False
            cols = [desc[0] for desc in cur.description]
            result = dict(zip(cols, row))
            if original_citation:
                return {original_citation: result}
            return result

    # -------------------------
    # Cleanup: close pool and main conn
    # -------------------------
    def close(self) -> None:
        try:
            if getattr(self, "pool", None):
                self.pool.closeall()
        except Exception:
            pass
        try:
            if getattr(self, "conn", None):
                self.conn.close()
        except Exception:
            pass


def run_searches_multithreaded(searcher: ArxivMetaSearchDB, queries_to_run: List[Dict[str, Any]], max_workers: int = 8) -> Dict[str, Any]:
    """
    queries_to_run: list of dicts with keys:
        - original_citation (original string)
        - glued_normalized_title
        - normalized_authors (list of normalized author strings) or None
    """
    results = []
    # start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_query = {ex.submit(searcher.search_paper, q["original_citation"], q["glued_normalized_title"], q.get("normalized_authors")): q for q in queries_to_run}
        # for fut in tqdm(as_completed(future_to_query), total=len(future_to_query), desc="Searching", unit="q"):
        for fut in as_completed(future_to_query):
            q = future_to_query[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"error": str(e), "query": q}
            # print(res)
            # print("-" * 80)
            if not any(value is False for value in res.values()):
                results.append(res)

    # elapsed = time.time() - start
    # print(f"Completed {len(results)} searches in {elapsed:.2f}s")
    result = {k: v for d in results for k, v in d.items()}
    return result


if __name__ == "__main__":
    db_worder = ArxivMetaSearchDB(PG)
    db_worder.build_db(arxiv_meta_path, dry_run=False)