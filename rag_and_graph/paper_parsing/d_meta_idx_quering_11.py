import json
from math import ceil
import psycopg2
from psycopg2.extras import execute_batch
from psycopg2.extras import RealDictCursor
import time
from datetime import datetime
from dateutil import parser as date_parser
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class ArxivMetaSearchDB:
    def __init__(self, PG):
        self.conn = psycopg2.connect(**PG)

    def _create_schema(self):
        create_arxiv_paper_query = '''
        CREATE TABLE IF NOT EXISTS PAPERS (
            id SERIAL PRIMARY KEY,
            title_and_authors TEXT NOT NULL,
            authors_str TEXT,
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
        Ensure pg_trgm + unaccent exist, add a title_norm column, keep it updated
        via trigger, and create a GIN trigram index on title_norm using CONCURRENTLY.

        Reason: unaccent() is not IMMUTABLE, so expressions like lower(unaccent(title))
        cannot be put directly into an index. We materialize the normalized title
        into a column and index that column instead.
        """
        # 1) Create extensions inside a transaction
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
        self.conn.commit()

        # 2) Add a materialized-normalized column if not exists
        with self.conn.cursor() as cur:
            cur.execute("""
            ALTER TABLE papers
            ADD COLUMN IF NOT EXISTS title_norm TEXT;
            """)
        self.conn.commit()

        # 3) Populate title_norm for existing rows (only update when needed)
        with self.conn.cursor() as cur:
            # use IS DISTINCT FROM to avoid updating identical values unnecessarily
            cur.execute("""
            UPDATE papers
            SET title_norm = lower(unaccent(title))
            WHERE title IS NOT NULL
            AND (title_norm IS NULL OR title_norm IS DISTINCT FROM lower(unaccent(title)));
            """)
        self.conn.commit()

        # 4) Create trigger function that keeps title_norm updated on INSERT/UPDATE
        #    Use CREATE OR REPLACE FUNCTION so re-running is safe.
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE OR REPLACE FUNCTION papers_title_norm_trigger()
            RETURNS trigger AS $$
            BEGIN
                -- If title is NULL, keep title_norm NULL
                IF NEW.title IS NULL THEN
                    NEW.title_norm := NULL;
                ELSE
                    NEW.title_norm := lower(unaccent(NEW.title));
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """)
        self.conn.commit()

        # 5) Create trigger (drop existing trigger first to be idempotent)
        with self.conn.cursor() as cur:
            cur.execute("""
            DROP TRIGGER IF EXISTS trg_papers_title_norm ON papers;
            CREATE TRIGGER trg_papers_title_norm
            BEFORE INSERT OR UPDATE OF title
            ON papers
            FOR EACH ROW
            EXECUTE FUNCTION papers_title_norm_trigger();
            """)
        self.conn.commit()

        # 6) Create the GIN trigram index on the materialized column using CONCURRENTLY
        # CREATE INDEX CONCURRENTLY cannot run inside a transaction block, so enable autocommit temporarily.
        prev_autocommit = getattr(self.conn, "autocommit", False)
        self.conn.autocommit = True
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_title_norm_trgm
                ON papers USING gin (title_norm gin_trgm_ops);
                """)
        finally:
            self.conn.autocommit = prev_autocommit

        # Optional: vacuum analyze to update planner statistics (good after big updates)
        with self.conn.cursor() as cur:
            cur.execute("ANALYZE papers;")
        self.conn.commit()

    # Insert all papers (even with duplicates)
        # Make columns: title_and_authors: str, authors_str: str, created:..., oai_header: jsonb, metadata: jsonb
    # Remove duplicates by title_and_authors -> only latest by created date remain
    def _insert_json(self, ndjson_path):
        def get_num_lines(file_path):
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)

        def format_authors(data):
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

        total_lines = get_num_lines(ndjson_path)
        with open(ndjson_path, 'r') as f:
            with self.conn.cursor() as cur:
                for line in tqdm(f, total=total_lines):
                    paper = json.loads(line)
                    authors = sorted(format_authors(paper['metadata'].get('authors', {}).get('author', [])))
                    authors_str = ', '.join(authors)
                    title = paper['metadata'].get('title', '')
                    title_and_authors = title + " | " + authors_str
                    oai_header = json.dumps(paper.get('oai_header', {}))
                    metadata = json.dumps(paper.get('metadata', {}))
                    created_date_str = paper['metadata'].get('created', '')
                    created_date = None
                    if created_date_str:
                        created_date = datetime.strptime(created_date_str, '%Y-%m-%d').date()

                    insert_query = '''
                    INSERT INTO PAPERS (title_and_authors, authors_str, created_date, oai_header, metadata)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                    '''
                    cur.execute(insert_query, (title_and_authors, authors_str, created_date, oai_header, metadata))
        self.conn.commit()

    def remove_duplicate_papers(self, dry_run=True, batch_size=1000):
        """
        Remove duplicate rows in PAPERS by title_and_authors keeping only the latest by created_date.
        - dry_run=True: will only return the number of rows that WOULD be deleted and show a preview.
        - dry_run=False: will actually delete rows in batches of `batch_size` and return number deleted.

        Behavior notes:
        - "Latest" is determined by created_date DESC (NULL treated as older because of NULLS LAST).
        - If created_date ties, the row with the highest id is kept (ORDER BY created_date DESC NULLS LAST, id DESC).
        """
        # 1) Preview how many duplicates exist
        count_query = """
        SELECT COUNT(*) FROM (
        SELECT ROW_NUMBER() OVER (
            PARTITION BY title_and_authors
            ORDER BY created_date DESC NULLS LAST, id DESC
        ) AS rn
        FROM PAPERS
        ) t
        WHERE t.rn > 1;
        """

        # 2) Query that returns ids we want to delete (those with rn > 1)
        ids_query = """
        SELECT id FROM (
        SELECT id,
                ROW_NUMBER() OVER (
                PARTITION BY title_and_authors
                ORDER BY created_date DESC NULLS LAST, id DESC
                ) AS rn
        FROM PAPERS
        ) t
        WHERE t.rn > 1;
        """

        with self.conn.cursor() as cur:
            cur.execute(count_query)
            will_delete_count = cur.fetchone()[0] # type: ignore

            if dry_run:
                # Show preview info and return the count (no changes)
                print(f"Duplicate groups found. Rows that would be removed: {will_delete_count}")
                # Return a small preview of duplicates (ids + title_and_authors + created_date)
                preview_detail_query = """
                SELECT id, title_and_authors, created_date FROM (
                SELECT id, title_and_authors, created_date,
                        ROW_NUMBER() OVER (
                        PARTITION BY title_and_authors
                        ORDER BY created_date DESC NULLS LAST, id DESC
                        ) AS rn
                FROM PAPERS
                ) t
                WHERE t.rn > 1
                LIMIT 50;  -- preview up to 50 rows
                """
                cur.execute(preview_detail_query)
                preview_rows = cur.fetchall()
                for r in preview_rows:
                    print(r)
                return {"would_delete": will_delete_count, "preview_rows": preview_rows}

            # Actual deletion path
            if will_delete_count == 0:
                print("No duplicates found. Nothing to delete.")
                return {"deleted": 0}

            # Fetch all ids to delete
            cur.execute(ids_query)
            all_ids = [row[0] for row in cur.fetchall()]

        # Delete in batches to avoid huge transactions / locks
        total_deleted = 0
        num_batches = ceil(len(all_ids) / batch_size)
        for i in tqdm(range(num_batches), desc="Deleting duplicate batches"):
            start = i * batch_size
            end = start + batch_size
            chunk = all_ids[start:end]
            if not chunk:
                continue
            with self.conn.cursor() as cur:
                # Use a parameterized IN clause via tuple
                cur.execute("DELETE FROM PAPERS WHERE id = ANY(%s);", (chunk,))
                deleted_this_chunk = cur.rowcount
                total_deleted += deleted_this_chunk
                # commit incrementally per batch
                self.conn.commit()

        self.conn.commit()

        print(f"Deleted {total_deleted} duplicate rows (kept latest per title_and_authors).")
        return {"deleted": total_deleted}

    def _vacuum_table(self):
        # Optional: VACUUM the table afterwards to reclaim space
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE PAPERS;")
        self.conn.autocommit = False

    def build_db(self, ndjson_path, dry_run):
        self._create_schema()
        self._insert_json(ndjson_path)
        self.remove_duplicate_papers(dry_run)
        self._vacuum_table
        

    def search_by_title(self, query: str, threshold: float = 0.95, limit: int = 5):
        """
        Search for a paper by title.
        - First tries exact match (case- and accent-insensitive).
        - If none, performs trigram similarity lookup and returns matches only
          if the top match has similarity >= threshold.
        Returns:
          - a single dict for exact match,
          - a list of dicts (ordered by similarity desc) if fuzzy matches found and top >= threshold,
          - False if no match meets the threshold.
        """
        q = query.strip()
        if not q:
            return False

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1) exact, case- and accent-insensitive match
            cur.execute(
                """
                SELECT *, 1.0 AS sim
                FROM papers
                WHERE lower(unaccent(title)) = lower(unaccent(%s))
                LIMIT 1;
                """,
                (q,)
            )
            exact = cur.fetchone()
            if exact:
                return exact

            # 2) fuzzy match using similarity()
            # We compute similarity on normalized form lower(unaccent(title)).
            cur.execute(
                """
                SELECT id, title,
                       similarity(lower(unaccent(title)), lower(unaccent(%s))) AS sim
                FROM papers
                WHERE similarity(lower(unaccent(title)), lower(unaccent(%s))) >= %s
                ORDER BY sim DESC
                LIMIT %s;
                """,
                (q, q, threshold, limit)
            )
            rows = cur.fetchall()
            if not rows:
                return False

            # rows is a list of dicts sorted by sim desc
            top = rows[0]
            if top["sim"] >= threshold:
                return rows
            return False


# Usage example
if __name__ == "__main__":
    ndjson_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson"
    PG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "arxiv_meta_db_2",
    "user": "postgres",
    "password": "@Q_Fa;ml$f!@94r"
}
    # searcher = PaperSearch("dbname=arxiv_meta_db_2 user=postgres password=@Q_Fa;ml$f!@94r")
    searcher = ArxivMetaSearchDB(PG)
    searcher.build_db(ndjson_path, dry_run=False)
    sys.exit()

    # Search examples
    query = "Generating text with recurrent neural networks"
    results = searcher.search_by_title(query)
    print(results)
    if results:
        for res in results:
            print(res)
            print('-'*100)


# if __name__ == "__main__":
    # Example paths
    # ndjson_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson"

    # PG = {
    #     "host": "localhost",
    #     "port": 5432,
    #     "dbname": "arxiv_meta_db_2",
    #     "user": "postgres",
    #     "password": "@Q_Fa;ml$f!@94r"
    # }
    # build_index("path/to/your/arxiv_data.ndjson", PG)

    # queries = [
    #     {'title': 'A Scalable Hierarchical Distributed Language Model', 'authors_teiled': ['A Mnih', 'G Hinton']},
    #     {'title': 'Gradient Flow in Recurrent Nets: the Diﬃculty of Learning Long-term Dependencies', 'authors_teiled': ['S Hochreiter', 'Y Bengio', 'P Frasconi', 'J Schmidhuber', 'S C Kremer', 'J F Kolen']},
    #     {'title': 'Lecture 6.5 - rmsprop: Divide the gradient by a running average of its recent magnitude', 'authors_teiled': ['T Tieleman', 'G Hinton']},
    #     {'title': 'A machine learning perspective on predictive coding with paq', 'authors_teiled': ['B Knoll', 'N De Freitas']},
    #     {'title': 'Data compression using adaptive cod- ing and partial string matching', 'authors_teiled': ['J G Cleary', 'Ian', 'I H Witten']},
    #     {'title': 'Better generative models for sequential data problems: Bidi- rectional recurrent mixture density networks', 'authors_teiled': ['M Schuster']},
    #     {'title': 'Subword language modeling with neural networks', 'authors_teiled': ['T Mikolov', 'I Sutskever', 'A Deoras', 'H Le', 'S Kombrink', 'J Cernocky']},
    #     {'title': 'Framewise phoneme classiﬁcation with bidi- rectional LSTM and other neural network architectures', 'authors_teiled': ['A Graves', 'J Schmidhuber']},
    #     {'title': 'The tagged LOB corpus user’s manual', 'authors_teiled': ['S Johansson', 'R Atwell', 'R Garside', 'G Leech']},
    #     {'title': 'The recurrent temporal restricted boltzmann machine', 'authors_teiled': ['I Sutskever', 'G E Hinton', 'G W Taylor']},
    #     {'title': 'The Minimum Description Length Principle (Adaptive Computation and Machine Learning)', 'authors_teiled': ['P D Gr¨unwald']},
    #     {'title': 'Oﬄine handwriting recognition with multi- dimensional recurrent neural networks', 'authors_teiled': ['A Graves', 'J Schmidhuber']},
    #     {'title': 'Generating text with recurrent neural networks', 'authors_teiled': ['I Sutskever', 'J Martens', 'G Hinton']},
    #     {'title': 'A ﬁrst look at music composition using lstm recurrent neural networks', 'authors_teiled': ['D Eck', 'J Schmidhuber']},
    #     {'title': 'Practical variational inference for neural networks', 'authors_teiled': ['A Graves']},
    #     {'title': 'Statistical Language Models based on Neural Networks', 'authors_teiled': ['T Mikolov']},
    #     {'title': 'Building a large annotated corpus of english: The penn treebank', 'authors_teiled': ['M P Marcus', 'B Santorini', 'M A Marcinkiewicz']},
    #     {'title': 'A Practical Guide to Training Restricted Boltzmann Machines', 'authors_teiled': ['G Hinton']},
    #     {'title': 'Low- rank matrix factorization for deep neural network training with high- dimensional output targets', 'authors_teiled': ['T N Sainath', 'A Mohamed', 'B Kingsbury', 'B Ramabhadran']},
    #     {'title': 'Long Short-Term Memory', 'authors_teiled': ['S Hochreiter', 'J Schmidhuber']},
    #     {'title': 'Speech recognition with deep recurrent neural networks', 'authors_teiled': ['A Graves', 'A Mohamed', 'G Hinton']},
    #     {'title': 'Learning precise timing with LSTM recurrent networks', 'authors_teiled': ['F Gers', 'N Schraudolph', 'J Schmidhuber']},
    #     {'title': 'IAM-OnDB - an on-line English sentence database acquired from handwritten text on a whiteboard', 'authors_teiled': ['M Liwicki', 'H Bunke']},
    #     {'title': 'Gradient-based learning algorithms for recur- rent networks and their computational complexity', 'authors_teiled': ['R Williams', 'D Zipser']},
    #     {'title': 'The Human Knowledge Compression Contest', 'authors_teiled': ['M Hutter']},
    #     {'title': 'Factored conditional restricted boltzmann machines for modeling motion style', 'authors_teiled': ['G W Taylor', 'G E Hinton']},
    #     {'title': 'Learning long-term dependencies with gradient descent is diﬃcult', 'authors_teiled': ['Y Bengio', 'P Simard', 'P Frasconi']},
    #     {'title': 'Modeling tempo- ral dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription', 'authors_teiled': ['N Boulanger-Lewandowski', 'Y Bengio', 'P Vincent']},
    #     {'title': 'Neural Networks for Pattern Recognition', 'authors_teiled': ['C Bishop']},
    #     {'title': 'An analysis of noise in recurrent neural networks: convergence and generalization', 'authors_teiled': ['K.-C Jim', 'C Giles', 'B Horne']},
    #     {'title': 'Sequence transduction with recurrent neural networks', 'authors_teiled': ['A Graves']},
    #     {'title': 'A fast and simple algorithm for training neural probabilistic language models', 'authors_teiled': ['A Mnih', 'Y W Teh']},
    #     {'title': 'Mixture density networks', 'authors_teiled': ['C Bishop']},
    # ]

    # start_time = time.time()
    
    # # Perform parallel search
    # parallel_results = search_papers_parallel(queries, PG)
    # print(parallel_results)
    # # Process results
    # for i, res in enumerate(parallel_results):
    #     if res:
    #         print(i+1, f"[TRUE TITLE]={res[0]['orig_title']} -", f"{res[0]['arxiv_id'], res[0]['title'], res[0]['authors']},") # type: ignore
    #     else:
    #         print(i, "[NOT FOUND]")
    #     # print("-"*100)

    # end_time = time.time()

    # print(f"Total time taken for the loop: {end_time - start_time} seconds")