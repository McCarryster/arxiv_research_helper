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
            arxiv_id TEXT NOT NULL UNIQUE,
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
        Create the pg_trgm extension (if missing) and an expression GIN index on
        lower(title_and_authors) to allow fast case-insensitive trigram similarity searches.
        """
        with self.conn.cursor() as cur:
            # Ensure the trigram extension is available
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

            # Create an expression GIN index on lower(title_and_authors)
            # This supports case-insensitive similarity searches and the % operator.
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_title_authors_trgm
            ON PAPERS USING gin (lower(title_and_authors) gin_trgm_ops);
            """)

        self.conn.commit()
        print("table papers indexed successfully")

    def check_index(self):
        """
        Return info about indexes for the PAPERS table and whether pg_trgm exists.
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'pg_trgm';")
            ext = cur.fetchone()

            cur.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'papers';
            """)
            indexes = cur.fetchall()

        return {
            "pg_trgm_installed": bool(ext),
            "indexes": indexes
        }

    def explain_search_plan(self, sample_query):
        """
        Return the planner's JSON EXPLAIN for the query that the search method runs.
        Useful to verify the planner chooses an index scan (GIN/trigram) or not.
        """
        if not sample_query:
            raise ValueError("sample_query must be provided")

        # Note: we escape the trigram operator '%' as '%%' in the Python string so the SQL
        # actually contains a single '%' when sent to Postgres.
        explain_sql = """
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT id
        FROM PAPERS
        WHERE lower(title_and_authors) %% lower(%s)
        ORDER BY similarity(lower(title_and_authors), lower(%s)) DESC
        LIMIT 1;
        """
        with self.conn.cursor() as cur:
            cur.execute(explain_sql, (sample_query, sample_query))
            plan = cur.fetchone()[0]  # JSON plan returned as text/json by psycopg2 # type: ignore

        return plan

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

    def vacuum_table(self):
        # Optional: VACUUM the table to reclaim space
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE PAPERS;")
        self.conn.autocommit = False
        print("vacuum completed on papers")

    def build_db(self, ndjson_path, dry_run):
        self._create_schema()
        self._insert_json(ndjson_path)
        self.remove_duplicate_papers(dry_run)
        self.vacuum_table()
        self._create_index()

    def search_title_and_authors(self, query, threshold=0.95):
        """
        Search PAPERS.title_and_authors for the single best match.
        Returns:
            - dict-like row (id, title_and_authors, authors_str, created_date, sim) if match >= threshold
            - False otherwise

        Important fix: the trigram operator `%` must be escaped in the Python string
        as `%%` so psycopg2 doesn't treat it as a Python-format placeholder.
        """
        if not query:
            return False

        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # set_limit affects the behavior of the % operator and can help the planner use the index
            cur.execute("SELECT set_limit(%s);", (float(threshold),))

            # NOTE: the trigram operator `%` is escaped as `%%` in the Python string
            cur.execute(
                """
                SELECT
                    id,
                    title_and_authors,
                    authors_str,
                    created_date,
                    similarity(lower(title_and_authors), lower(%s)) AS sim
                FROM PAPERS
                WHERE lower(title_and_authors) %% lower(%s)
                ORDER BY sim DESC
                LIMIT 1;
                """,
                (query, query)
            )
            row = cur.fetchone()

        if not row:
            return False

        sim_val = float(row.get("sim", 0.0))
        if sim_val < float(threshold):
            return False

        return row

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
    searcher = ArxivMetaSearchDB(PG)
    # query = "TRANSIMS traffic flow characteristics | Barrett Christopher L., Donnelly Rick, Nagel Kai, Pieck Martin, Stretz Paula"
    # query = "Sequence transduction with recurrent neural networks | A Graves"
    # result = searcher.search_title_and_authors(query, threshold=0.85)
    # if result:
    #     print("Found:", result)
    # else:
    #     print("No close match (>= 0.95) found")

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

    queries = [
        {'title': 'A decomposable attention model', 'authors_teiled': ['Ankur Parikh', 'Oscar Täckström', 'Dipanjan Das', 'Jakob Uszkoreit']},
        {'title': 'End-to-end memory networks', 'authors_teiled': ['Sainbayar Sukhbaatar', 'Arthur Szlam', 'Jason Weston', 'Rob Fergus', 'C Cortes', 'N D Lawrence', 'D D Lee', 'M Sugiyama', 'R Garnett']},
        {'title': 'Neural GPUs learn algorithms', 'authors_teiled': ['Łukasz Kaiser', 'Ilya Sutskever']},
        {'title': 'Factorization tricks for LSTM networks', 'authors_teiled': ['Oleksii Kuchaiev', 'Boris Ginsburg']},
        {'title': 'Long short-term memory', 'authors_teiled': ['Sepp Hochreiter', 'Jürgen Schmidhuber']},
        {'title': 'Structured attention networks', 'authors_teiled': ['Yoon Kim', 'Carl Denton', 'Luong Hoang', 'Alexander M Rush']},
        {'title': 'Rethinking the inception architecture for computer vision', 'authors_teiled': ['Christian Szegedy', 'Vincent Vanhoucke', 'Sergey Ioffe', 'Jonathon Shlens', 'Zbigniew Wojna']},
        {'title': 'Empirical evaluation of gated recurrent neural networks on sequence modeling', 'authors_teiled': ['Junyoung Chung', 'Çaglar Gülçehre', 'Kyunghyun Cho', 'Yoshua Bengio']},
        {'title': 'Effective approaches to attention- based neural machine translation', 'authors_teiled': ['Minh-Thang Luong', 'Hieu Pham', 'Christopher D Manning']},
        {'title': 'Neural machine translation in linear time', 'authors_teiled': ['Nal Kalchbrenner', 'Lasse Espeholt', 'Karen Simonyan', 'Aaron Van Den Oord', 'Alex Graves', 'Ko- Ray Kavukcuoglu']},
        {'title': 'Learning phrase representations using rnn encoder-decoder for statistical machine translation', 'authors_teiled': ['Kyunghyun Cho', 'Bart Van Merrienboer', 'Caglar Gulcehre', 'Fethi Bougares', 'Holger Schwenk', 'Yoshua Bengio']},
        {'title': 'Using the output embedding to improve language models', 'authors_teiled': ['Ofir Press', 'Lior Wolf']},
        {'title': 'A deep reinforced model for abstractive summarization', 'authors_teiled': ['Romain Paulus', 'Caiming Xiong', 'Richard Socher']},
        {'title': 'A structured self-attentive sentence embedding', 'authors_teiled': ['Zhouhan Lin', 'Minwei Feng', 'Cicero Nogueira Dos Santos', 'Mo Yu', 'Bing Xiang', 'Bowen Zhou', 'Yoshua Bengio']},
        {'title': 'Neural machine translation of rare words with subword units', 'authors_teiled': ['Rico Sennrich', 'Barry Haddow', 'Alexandra Birch']},
        {'title': 'Building a large annotated corpus of english: The penn treebank', 'authors_teiled': ['Mary Mitchell P Marcus', 'Ann Marcinkiewicz', 'Beatrice Santorini']},
        {'title': 'Layer normalization', 'authors_teiled': ['Jimmy Lei Ba', 'Jamie Ryan Kiros', 'Geoffrey E Hinton']},
        {'title': 'Xception: Deep learning with depthwise separable convolutions', 'authors_teiled': ['Francois Chollet']},
        {'title': 'Convolu- tional sequence to sequence learning', 'authors_teiled': ['Jonas Gehring', 'Michael Auli', 'David Grangier', 'Denis Yarats', 'Yann N Dauphin']},
        {'title': 'Generating sequences with recurrent neural networks', 'authors_teiled': ['Alex Graves']},
        {'title': 'Outrageously large neural networks: The sparsely-gated mixture-of-experts layer', 'authors_teiled': ['Noam Shazeer', 'Azalia Mirhoseini', 'Krzysztof Maziarz', 'Andy Davis', 'Quoc Le', 'Geoffrey Hinton', 'Jeff Dean']},
        {'title': 'Deep residual learning for im- age recognition', 'authors_teiled': ['Kaiming He', 'Xiangyu Zhang', 'Shaoqing Ren', 'Jian Sun']},
        {'title': 'Adam: A method for stochastic optimization', 'authors_teiled': ['Diederik Kingma', 'Jimmy Ba']},
        {'title': 'Can active memory replace attention?', 'authors_teiled': ['Łukasz Kaiser', 'Samy Bengio']},
        {'title': 'Long short-term memory-networks for machine reading', 'authors_teiled': ['Jianpeng Cheng', 'Li Dong', 'Mirella Lapata']},
        {'title': 'Learning accurate, compact, and interpretable tree annotation', 'authors_teiled': ['Slav Petrov', 'Leon Barrett', 'Romain Thibaux', 'Dan Klein']},
        {'title': 'Dropout: a simple way to prevent neural networks from overfitting', 'authors_teiled': ['Nitish Srivastava', 'Geoffrey E Hinton', 'Alex Krizhevsky', 'Ilya Sutskever', 'Ruslan Salakhutdi- Nov']},
        {'title': 'Grammar as a foreign language', 'authors_teiled': ['Vinyals', 'Koo Kaiser', 'Petrov', 'Sutskever', 'Hinton']},
        {'title': 'Google’s neural machine translation system: Bridging the gap between human and machine translation', 'authors_teiled': ['Yonghui Wu', 'Mike Schuster', 'Zhifeng Chen', 'V Quoc', 'Mohammad Le', 'Wolfgang Norouzi', 'Maxim Macherey', 'Yuan Krikun', 'Qin Cao', 'Klaus Gao', 'Macherey']},
        {'title': 'Exploring the limits of language modeling', 'authors_teiled': ['Rafal Jozefowicz', 'Oriol Vinyals', 'Mike Schuster', 'Noam Shazeer', 'Yonghui Wu']},
        {'title': 'Sequence to sequence learning with neural networks', 'authors_teiled': ['Ilya Sutskever', 'Oriol Vinyals', 'Quoc Vv Le']},
        {'title': 'Neural machine translation by jointly learning to align and translate', 'authors_teiled': ['Dzmitry Bahdanau', 'Kyunghyun Cho', 'Yoshua Bengio']},
        {'title': 'Recurrent neural network grammars', 'authors_teiled': ['Chris Dyer', 'Adhiguna Kuncoro', 'Miguel Ballesteros', 'Noah A Smith']},
        {'title': 'Gradient flow in recurrent nets: the difficulty of learning long-term dependencies', 'authors_teiled': ['Sepp Hochreiter', 'Yoshua Bengio', 'Paolo Frasconi', 'Jürgen Schmidhuber']},
        {'title': 'Massive exploration of neural machine translation architectures', 'authors_teiled': ['Denny Britz', 'Anna Goldie', 'Minh-Thang Luong', 'V Quoc', 'Le']},
    ]

    # query = "End-To-End Memory Networks"
    # print(searcher.search_title_and_authors(query, threshold=0.2))


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

    formatted_queries = []
    for q in queries:
        authors = q['authors_teiled']
        # If authors are dicts, use format_authors; else, leave as-is
        if authors and isinstance(authors[0], dict):
            authors_str = ', '.join(format_authors(authors))
        else:
            authors_str = ', '.join(authors)
        formatted_queries.append(f"{q['title']} | {authors_str}")

    for que in formatted_queries:
        print(que, "|", searcher.search_title_and_authors(que, threshold=0.6))
        print('-'*100)
    # FIX: IF TOO MANY AUTHORS IN ENTRY FROM DB THEN IN QUERY -> RETURNS False, BUT SHOULD BE TRUE BECAUSE TITLE MATCH AND SOME OF THE AUTHORS MATCH


    # formatted_queries = [
    #     f"{q['title']} | {', '.join(q['authors_teiled'])}" for q in queries
    # ]

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