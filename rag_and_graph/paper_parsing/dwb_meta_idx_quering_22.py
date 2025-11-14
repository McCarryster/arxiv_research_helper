import psycopg2
import psycopg2.extras
import psycopg2.pool
import json
import unicodedata
import re
import csv
from datetime import datetime
from typing import List, Dict, Any, Set, Union
import psycopg2.extras
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class ArxivMetaSearchDB:
    def __init__(self, PG: dict, pool_minconn: int = 1, pool_maxconn: int = 10):
        """
        PG: dict with psycopg2.connect parameters, e.g. {"host":..., "dbname":..., "user":..., "password":...}
        pool_minconn/pool_maxconn: controls the threaded connection pool size.
        """
        # Keep one connection for single-threaded DB ops (schema creation, inserts, etc.)
        self.conn = psycopg2.connect(**PG)
        # Thread-safe connection pool for multi-threaded searches.
        # minconn should be at least 1, maxconn should be >= number of worker threads you plan to use.
        self.pool = psycopg2.pool.ThreadedConnectionPool(pool_minconn, pool_maxconn, **PG)

    # -------------------------
    # Normalization utilities
    # -------------------------
    def normalize_text(self, title: str) -> str:
        # Step 1: Unicode normalization to remove accents
        normalized = unicodedata.normalize('NFKD', title)
        normalized = normalized.encode('ASCII', 'ignore').decode('utf-8')
        # Step 2: Lowercase
        normalized = normalized.lower()
        # Step 3: Remove unwanted characters (keep letters, numbers, and spaces)
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        # Step 4: Replace multiple spaces with one
        normalized = re.sub(r'\s+', ' ', normalized)
        # Step 5: Strip whitespace and make lowercase
        normalized = normalized.strip()
        normalized = normalized.lower()
        return normalized

    def format_authors(self, data):
        if isinstance(data, dict):
            data = [data]
        result = []
        for item in data:
            keyname = item.get('keyname', '')
            forenames = item.get('forenames', '')
            if forenames:
                result.append(f"{keyname} {forenames}")
            else:
                result.append(keyname)
        return result

    def normalize_author_list(self, authors: List[str]) -> List[str]:
        if authors is None:
            return []
        return [self.normalize_text(a) for a in authors]

    # -------------------------
    # Schema + index creation
    # -------------------------
    def _create_schema(self):
        create_arxiv_paper_query = '''
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
        '''
        with self.conn.cursor() as cur:
            cur.execute(create_arxiv_paper_query)
        self.conn.commit()

    def _create_index(self):
        """
        Create indexes to speed lookups by PAPERS.glued_normalized_title.

        - Creates a B-tree index for fast exact-equality searches (glued_normalized_title = '...').
        - Attempts to create the pg_trgm extension and a GIN trigram index for fast ILIKE / fuzzy
        / similarity searches (e.g. WHERE glued_normalized_title ILIKE '%grammar%').
        If the extension/index creation fails (permissions, extension unavailable), it will
        print a warning and continue — the equality index will still be created.
        """
        with self.conn.cursor() as cur:
            # B-tree index for fast equality lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_glued_normalized_title
                ON PAPERS (glued_normalized_title);
            """)

            # Try to create trigram extension + GIN trigram index for fast ILIKE / similarity.
            # This is optional — if the DB user can't create extensions this will raise and be caught.
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_papers_glued_normalized_title_trgm
                    ON PAPERS USING gin (glued_normalized_title gin_trgm_ops);
                """)
            except Exception as e:
                # Don't fail the whole operation if pg_trgm isn't installable or the index can't be created.
                # In production you might want to log this differently.
                print("Warning: could not create trigram index or extension:", e)
        self.conn.commit()
        print('Index created successfully')

    def check_index(self):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('pg_trgm', 'unaccent');")
            exts = cur.fetchall()
            cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'papers';")
            indexes = cur.fetchall()
        print({"extensions_installed": exts, "indexes": indexes})
        return {"extensions_installed": exts, "indexes": indexes}

    # -------------------------
    # Insert NDJSON
    # -------------------------
    def _insert_ndjson(self, ndjson_path):
        def get_num_lines(file_path):
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)

        total_lines = get_num_lines(ndjson_path)
        with open(ndjson_path, 'r') as f:
            with self.conn.cursor() as cur:
                for line in tqdm(f, total=total_lines):
                    paper = json.loads(line)

                    arxiv_id = paper['metadata'].get('id', None)

                    title = paper['metadata'].get('title', '')
                    normalized_title = self.normalize_text(title)
                    glued_normalized_title = normalized_title.replace(" ", "")

                    authors = sorted(self.format_authors(paper['metadata'].get('authors', {}).get('author', [])))
                    normalized_authors = self.normalize_author_list(authors)
                    authors_str = ', '.join(normalized_authors)

                    title_and_authors = title + " | " + authors_str

                    created_date_str = paper['metadata'].get('created', '')
                    created_date = None
                    if created_date_str:
                        created_date = datetime.strptime(created_date_str, '%Y-%m-%d').date()

                    oai_header = json.dumps(paper.get('oai_header', {}))
                    metadata = json.dumps(paper.get('metadata', {}))

                    insert_query = '''
                    INSERT INTO PAPERS (arxiv_id, title, normalized_title, glued_normalized_title, normalized_authors, title_and_authors, created_date, oai_header, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (arxiv_id) DO NOTHING;
                    '''
                    cur.execute(insert_query, (arxiv_id, title, normalized_title, glued_normalized_title, normalized_authors, title_and_authors, created_date, oai_header, metadata))
        self.conn.commit()

    # -------------------------
    # build_db orchestration
    # -------------------------
    def build_db(self, ndjson_path, dry_run=True):
        self._create_schema()
        insert_summary = self._insert_ndjson(ndjson_path)
        dup_summary = self.remove_duplicate_papers(dry_run=dry_run)
        self.vacuum_tables()
        self._create_index()
        idx_summary = self.check_index()
        return {"insert": insert_summary, "duplicates": dup_summary, "indexes": idx_summary}

    # -------------------------
    # Duplicate removal
    # -------------------------
    def remove_duplicate_papers(self, dry_run: bool = False) -> Dict[str, Any]:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
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
                """)
                dup_count = cur.fetchone()[0] # type: ignore

                if dup_count == 0:
                    return {"duplicates_found": 0, "deleted": 0, "deleted_ids": []}

                if dry_run:
                    return {"duplicates_found": dup_count, "deleted": 0, "deleted_ids": []}

                cur.execute("""
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
                """)
                deleted_rows = cur.fetchall()
                deleted_ids = [row[0] for row in deleted_rows]

            self.conn.commit()

            return {
                "duplicates_found": dup_count,
                "deleted": len(deleted_ids),
                "deleted_ids": deleted_ids
            }

        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            raise

    # -------------------------
    # VACUUM helper
    # -------------------------
    def vacuum_tables(self):
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
    # Main search function
    # -------------------------
    def search_paper(self, glued_normalized_title, query_authors=None):
        """
        Search for a paper by glued_normalized_title and refine results by authors.

        Behavior:
        1. If glued_normalized_title is empty -> return False.
        2. First, do a fast exact-title lookup (uses B-tree index on glued_normalized_title).
        - Fetch up to N_TITLE_CANDIDATES candidate rows (keeps the lookup fast while allowing
            multiple db rows with the same title).
        3. If exact-title candidates are found:
        - If query_authors is None: return the top candidate (by created_date).
        - If query_authors provided: compute an order-insensitive author match score and
            return the candidate with the highest positive score. If no candidate has a
            positive author match, return False.
        4. If no exact-title candidates found: perform a trigram-similarity title search
        (uses pg_trgm + GIN trigram index when available) to retrieve top candidates.
        Then apply the same author-refinement as above.
        5. If no title matches at all -> return False.
        6. If title matches exist but none match authors (when authors were provided) -> False.

        Parameters:
            glued_normalized_title (str): normalized / "glued" title to search for.
            query_authors (list[str] | None): list of normalized author strings for the query,
                e.g. ['ukasz kaiser', 'samy bengio']. Can be None.

        Returns:
            dict: column->value mapping for the selected top match, or False if no acceptable
                match was found.
        """
        if not glued_normalized_title:
            return False

        # Number of candidate rows to pull from the DB for author refinement.
        # Small number keeps queries very fast while still allowing meaningful author comparisons.
        N_TITLE_CANDIDATES = 50

        def _canonical_author(name):
            """
            Create an order-insensitive canonical form for an author name:
            - lower-cases
            - splits on whitespace
            - sorts tokens (so 'samy bengio' -> ['bengio','samy'] -> 'bengio samy')
            This allows matching 'firstname lastname' with 'lastname firstname' and small variations.
            """
            if not name:
                return ""
            # ensure it's a string
            s = str(name).strip().lower()
            if not s:
                return ""
            tokens = [t for t in s.split() if t]
            tokens.sort()
            return " ".join(tokens)

        def _normalize_db_authors_field(db_val):
            """
            Normalize how the DB delivered the normalized_authors field into a Python list of strings.

            psycopg2 usually gives SQL arrays as Python lists. If for some reason we received a string
            like '{"bengio samy","kaiser ukasz"}', try to parse it simply.
            """
            if db_val is None:
                return []
            if isinstance(db_val, (list, tuple)):
                return [str(x) for x in db_val]
            if isinstance(db_val, str):
                # crude parser for Postgres array text representation
                v = db_val.strip()
                if v.startswith("{") and v.endswith("}"):
                    v = v[1:-1]
                # split on commas that are not inside quotes
                parts = []
                cur = []
                in_quote = False
                i = 0
                while i < len(v):
                    ch = v[i]
                    if ch == '"' and (i == 0 or v[i-1] != '\\'):
                        in_quote = not in_quote
                    elif ch == ',' and not in_quote:
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
                # filter empties
                return [p for p in parts if p]
            # fallback
            return [str(db_val)]

        def _best_candidate_by_authors(rows, cols, query_authors_list):
            """
            Given DB candidate rows and the column names list, pick the best candidate that matches authors.
            Returns a dict (row) or None if no candidate has any positive author match.
            """
            if not query_authors_list:
                # if no query authors provided, return the first/top row
                if not rows:
                    return None
                return dict(zip(cols, rows[0]))

            # Canonicalize query authors to a set for fast membership checks
            canonical_query = set()
            for a in query_authors_list:
                ca = _canonical_author(a)
                if ca:
                    canonical_query.add(ca)

            if not canonical_query:
                # query authors were present but after canonicalization none remain meaningful
                # treat as "no authors provided" and return top candidate
                if not rows:
                    return None
                return dict(zip(cols, rows[0]))

            best_row = None
            best_score = 0
            best_created_date = None  # tie-breaker: prefer newer created_date if scores tie

            # find column index for normalized_authors and created_date
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
                # canonicalize db authors into a set
                canonical_db = set(_canonical_author(x) for x in db_auth_list if x)
                # compute intersection size
                score = len(canonical_query & canonical_db)
                if score > best_score:
                    best_score = score
                    best_row = r
                    if idx_date is not None:
                        best_created_date = r[idx_date]
                elif score == best_score and score > 0:
                    # tie-break: prefer newer created_date if available
                    if idx_date is not None and best_created_date is not None:
                        try:
                            if r[idx_date] and r[idx_date] > best_created_date:
                                best_row = r
                                best_created_date = r[idx_date]
                        except Exception:
                            # if comparison fails, ignore tie-break
                            pass

            if best_score > 0 and best_row is not None:
                return dict(zip(cols, best_row))
            return None

        with self.conn.cursor() as cur:
            # 1) Fast exact equality lookup (uses B-tree index)
            cur.execute(f"""
                SELECT
                    id,
                    arxiv_id,
                    title,
                    normalized_title,
                    normalized_authors,
                    title_and_authors,
                    created_date
                FROM PAPERS
                WHERE glued_normalized_title = %s
                ORDER BY created_date DESC
                LIMIT %s;
            """, (glued_normalized_title, N_TITLE_CANDIDATES))
            exact_rows = cur.fetchall()

            cols = [d[0] for d in cur.description]

            if exact_rows:
                # We have title matches. Now refine by authors (if any).
                best = _best_candidate_by_authors(exact_rows, cols, query_authors)
                if best:
                    return best
                # According to the requirements: if there is a match in title but not a single match in authors -> return False
                return False

            # 2) No exact title matches -> try trigram similarity fallback (if available)
            try:
                cur.execute(f"""
                    SELECT
                        id,
                        arxiv_id,
                        title,
                        normalized_title,
                        normalized_authors,
                        title_and_authors,
                        created_date
                    FROM PAPERS
                    WHERE glued_normalized_title % %s
                    ORDER BY similarity(glued_normalized_title, %s) DESC
                    LIMIT %s;
                """, (glued_normalized_title, glued_normalized_title, N_TITLE_CANDIDATES))
                trig_rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
            except Exception:
                # trigram not available or query failed -> no title matches
                return False

            if not trig_rows:
                return False

            # We have trigram title matches. Refine by authors (if any).
            best = _best_candidate_by_authors(trig_rows, cols, query_authors)
            if best:
                return best
            # Title found but no author matches
            return False

    # -------------------------
    # Cleanup: close pool and main conn
    # -------------------------
    def close(self):
        try:
            if self.pool:
                self.pool.closeall()
        except Exception:
            pass
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass


start = time.time()  # start the timer
if __name__ == "__main__":
    ndjson_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson"
    PG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "arxiv_meta_db_5",
    "user": "postgres",
    "password": "@Q_Fa;ml$f!@94r"
    }

    # Instantiate the searcher. Ensure pool_maxconn is at least as large as max_workers you'll use below.
    # Tune pool_maxconn to match your DB's allowed concurrent connections.
    searcher = ArxivMetaSearchDB(PG, pool_minconn=20, pool_maxconn=20)
    # searcher.build_db(ndjson_path, dry_run=False)

    # Prepare normalized queries for batch search
    queries = [
        {'title': 'Can active memory replace attention?', 'authors_teiled': ['Łukasz Kaiser', 'Samy Bengio']}, # True
        {'title': 'Deep residual learning for im- age recognition', 'authors_teiled': ['Kaiming He', 'Xiangyu Zhang', 'Shaoqing Ren', 'Jian Sun']}, # True
        {'title': 'Gradient flow in recurrent nets: the difficulty of learning long-term dependencies', 'authors_teiled': ['Sepp Hochreiter', 'Yoshua Bengio', 'Paolo Frasconi', 'Jürgen Schmidhuber']}, # False
        {'title': 'Adam: A method for stochastic optimization', 'authors_teiled': ['Diederik Kingma', 'Jimmy Ba']},
        {'title': 'Neural machine translation of rare words with subword units', 'authors_teiled': ['Rico Sennrich', 'Barry Haddow', 'Alexandra Birch']},
        {'title': 'Grammar as a foreign language', 'authors_teiled': ['Vinyals', 'Koo Kaiser', 'Petrov', 'Sutskever', 'Hinton']},
        {'title': 'Generating sequences with recurrent neural networks', 'authors_teiled': ['Alex Graves']},
        {'title': 'Using the output embedding to improve language models', 'authors_teiled': ['Ofir Press', 'Lior Wolf']},
        {'title': 'Convolu- tional sequence to sequence learning', 'authors_teiled': ['Jonas Gehring', 'Michael Auli', 'David Grangier', 'Denis Yarats', 'Yann N Dauphin']},
        {'title': 'Massive exploration of neural machine translation architectures', 'authors_teiled': ['Denny Britz', 'Anna Goldie', 'Minh-Thang Luong', 'V Quoc', 'Le']},
        {'title': 'Sequence to sequence learning with neural networks', 'authors_teiled': ['Ilya Sutskever', 'Oriol Vinyals', 'Quoc Vv Le']},
        {'title': 'Long short-term memory', 'authors_teiled': ['Sepp Hochreiter', 'Jürgen Schmidhuber']},
        {'title': 'Learning accurate, compact, and interpretable tree annotation', 'authors_teiled': ['Slav Petrov', 'Leon Barrett', 'Romain Thibaux', 'Dan Klein']},
        {'title': 'End-to-end memory networks', 'authors_teiled': ['Sainbayar Sukhbaatar', 'Arthur Szlam', 'Jason Weston', 'Rob Fergus', 'C Cortes', 'N D Lawrence', 'D D Lee', 'M Sugiyama', 'R Garnett']},
        {'title': 'Factorization tricks for LSTM networks', 'authors_teiled': ['Oleksii Kuchaiev', 'Boris Ginsburg']},
        {'title': 'A decomposable attention model', 'authors_teiled': ['Ankur Parikh', 'Oscar Täckström', 'Dipanjan Das', 'Jakob Uszkoreit']},
        {'title': 'Neural GPUs learn algorithms', 'authors_teiled': ['Łukasz Kaiser', 'Ilya Sutskever']},
        {'title': 'Google’s neural machine translation system: Bridging the gap between human and machine translation', 'authors_teiled': ['Yonghui Wu', 'Mike Schuster', 'Zhifeng Chen', 'V Quoc', 'Mohammad Le', 'Wolfgang Norouzi', 'Maxim Macherey', 'Yuan Krikun', 'Qin Cao', 'Klaus Gao', 'Macherey']},
        {'title': 'Effective approaches to attention- based neural machine translation', 'authors_teiled': ['Minh-Thang Luong', 'Hieu Pham', 'Christopher D Manning']},
        {'title': 'Long short-term memory-networks for machine reading', 'authors_teiled': ['Jianpeng Cheng', 'Li Dong', 'Mirella Lapata']},
        {'title': 'Neural machine translation by jointly learning to align and translate', 'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']},
        {'title': 'Outrageously large neural networks: The sparsely-gated mixture-of-experts layer', 'authors_teiled': ['Noam Shazeer', 'Azalia Mirhoseini', 'Krzysztof Maziarz', 'Andy Davis', 'Quoc Le', 'Geoffrey Hinton', 'Jeff Dean']},
        {'title': 'Exploring the limits of language modeling', 'authors_teiled': ['Rafal Jozefowicz', 'Oriol Vinyals', 'Mike Schuster', 'Noam Shazeer', 'Yonghui Wu']},
        {'title': 'A structured self-attentive sentence embedding', 'authors_teiled': ['Zhouhan Lin', 'Minwei Feng', 'Cicero Nogueira Dos Santos', 'Mo Yu', 'Bing Xiang', 'Bowen Zhou', 'Yoshua Bengio']},
        {'title': 'Empirical evaluation of gated recurrent neural networks on sequence modeling', 'authors_teiled': ['Junyoung Chung', 'Çaglar Gülçehre', 'Kyunghyun Cho', 'Yoshua Bengio']},
        {'title': 'Dropout: a simple way to prevent neural networks from overfitting', 'authors_teiled': ['Nitish Srivastava', 'Geoffrey E Hinton', 'Alex Krizhevsky', 'Ilya Sutskever', 'Ruslan Salakhutdi- Nov']},
        {'title': 'Layer normalization', 'authors_teiled': ['Jimmy Lei Ba', 'Jamie Ryan Kiros', 'Geoffrey E Hinton']},
        {'title': 'Recurrent neural network grammars', 'authors_teiled': ['Chris Dyer', 'Adhiguna Kuncoro', 'Miguel Ballesteros', 'Noah A Smith']},
        {'title': 'Rethinking the inception architecture for computer vision', 'authors_teiled': ['Christian Szegedy', 'Vincent Vanhoucke', 'Sergey Ioffe', 'Jonathon Shlens', 'Zbigniew Wojna']},
        {'title': 'Building a large annotated corpus of english: The penn treebank', 'authors_teiled': ['Mary Mitchell P Marcus', 'Ann Marcinkiewicz', 'Beatrice Santorini']},
        {'title': 'Neural machine translation in linear time', 'authors_teiled': ['Nal Kalchbrenner', 'Lasse Espeholt', 'Karen Simonyan', 'Aaron Van Den Oord', 'Alex Graves', 'Ko- Ray Kavukcuoglu']},
        {'title': 'Structured attention networks', 'authors_teiled': ['Yoon Kim', 'Carl Denton', 'Luong Hoang', 'Alexander M Rush']},
        {'title': 'Xception: Deep learning with depthwise separable convolutions', 'authors_teiled': ['Francois Chollet']},
        {'title': 'Learning phrase representations using rnn encoder-decoder for statistical machine translation', 'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merrienboer', 'Caglar Gulcehre', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio']},
        {'title': 'A deep reinforced model for abstractive summarization', 'authors_teiled': ['Romain Paulus', 'Caiming Xiong', 'Richard Socher']},
    ]
    queries_to_run = []
    for q in queries:
        title_raw = q.get('title', '')
        authors_raw = q.get('authors_teiled', [])
        normalized_title = searcher.normalize_text(title_raw)
        normalized_authors = searcher.normalize_author_list(authors_raw)
        queries_to_run.append({
            "normalized_title": normalized_title.replace(" ", ""),
            "normalized_authors": normalized_authors
        })

    for query in queries_to_run:
        print(query['normalized_title'])
        # print(query['normalized_authors'])
        # break
        # print(searcher.search_paper(query['normalized_title'], similarity_threshold=0.35))
        print(searcher.search_paper(query['normalized_title']))
        print('-'*100)