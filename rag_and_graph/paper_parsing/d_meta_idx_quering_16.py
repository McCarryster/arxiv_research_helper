import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.extras
from psycopg2.extras import execute_values
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import csv
import json
from datetime import datetime
import unicodedata
import re
import difflib
from math import ceil
from typing import List, Union, Dict, Set, Any
from typing import List, Union, Optional
import sys


# 1. Schema looks like that:
    # id SERIAL PRIMARY KEY, 
    # title TEXT NOT NULL,                  # Original title
    # normalized_title TEXT NOT NULL,       # Field to make searches on
    # normalized_authors TEXT[] NOT NULL,   # Field to make searches on
    # title_and_authors TEXT NOT NULL,      # Field to remove duplicates
    # created_date DATE NOT NULL,           # Field to help remove duplicates
    # oai_header JSONB DEFAULT '{}'::jsonb,
    # metadata JSONB DEFAULT '{}'::jsonb

# 2. Make remove_duplicate_papers() function that will find duplicates by title_and_authors and remove older by created_date, so only the latest by created_date entries should remain

# 3. Create index on normalized_title

# 4. For search:
#     1. Find exact (or close by 0.8 similarity) titles using query title (query title will also be normalized)
#     2. Refine the search by query authors (query authors will also be normalized). Take the highest search by the % of authors overlap (between query authors and normalized_authors)
#     3. Return only 1 result (the highest match only). If there is no match -> return False


class ArxivMetaSearchDB:
    def __init__(self, PG):
        """
        PG: dict with psycopg2.connect parameters, e.g. {"host":..., "dbname":..., "user":..., "password":...}
        """
        self.conn = psycopg2.connect(**PG)

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

    def _normalize_author_list(self, authors: List[str]) -> List[str]:
        if authors is None:
            return []
        return [self.normalize_text(a) for a in authors]

    def _author_to_token_list(self, author: str) -> List[str]:
        """
        Turn an author string into a list of normalized tokens (no particular order).
        Example: "Wu Yonghui" -> ["wu","yonghui"]
        """
        norm = self.normalize_text(author)
        if not norm:
            return []
        tokens = [t for t in norm.split() if t]
        return tokens

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

    def _create_index(self) -> None:
        """
        Create a trigram index on normalized_title using the pg_trgm extension for fast similarity searches.

        What this does:
        1. Ensures the pg_trgm extension is available in the current database.
        2. Creates a GIN index on the normalized_title column using gin_trgm_ops
           which speeds up similarity searches (operator: % and function: similarity()).
        """
        with self.conn.cursor() as cur:
            # Ensure pg_trgm is available
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            # Create a GIN trigram index (efficient for similarity / % operator)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_papers_normalized_title_trgm
                ON PAPERS
                USING gin (normalized_title gin_trgm_ops);
            """)
        self.conn.commit()

    def check_index(self):
        """Return whether pg_trgm and unaccent are present and list of indexes on PAPERS."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
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

                    arxiv_id = paper['metadata'].get('id', None)                                             # actual id of a paper

                    title = paper['metadata'].get('title', '')                                               # default title
                    normalized_title = self.normalize_text(title)                                            # normalized title for search

                    authors = sorted(self.format_authors(paper['metadata'].get('authors', {}).get('author', []))) # default authors
                    normalized_authors = self._normalize_author_list(authors)                                # normalized authors for search
                    authors_str = ', '.join(normalized_authors)                                              # merged by authors for duplicate removal

                    title_and_authors = title + " | " + authors_str                                          # field for duplicate removal

                    created_date_str = paper['metadata'].get('created', '')
                    created_date = None                                                                      # field to help remove duplicates
                    if created_date_str:
                        created_date = datetime.strptime(created_date_str, '%Y-%m-%d').date()

                    oai_header = json.dumps(paper.get('oai_header', {}))                                     # some shit idc
                    metadata = json.dumps(paper.get('metadata', {}))                                         # usefull metdata

                    insert_query = '''
                    INSERT INTO PAPERS (arxiv_id, title, normalized_title, normalized_authors, title_and_authors, created_date, oai_header, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    '''
                    cur.execute(insert_query, (arxiv_id, title, normalized_title, normalized_authors, title_and_authors, created_date, oai_header, metadata))
        self.conn.commit()


    # -------------------------
    # build_db orchestration
    # -------------------------
    def build_db(self, ndjson_path, dry_run=True):
        """
        Build DB: create schema, insert ndjson, remove duplicates, vacuum, create indexes, check index.
        """
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
        """
        Remove duplicate rows in PAPERS where duplicates are defined by identical title_and_authors.
        Keeps only the latest row per title_and_authors based on created_date.
        If multiple rows share the same (title_and_authors, created_date), keeps the row with the highest id.

        Parameters
        ----------
        dry_run : bool
            If True, no rows are deleted; method returns how many rows would be deleted.

        Returns
        -------
        dict
            A summary dictionary. Example:
            {
                "duplicates_found": <int>,
                "deleted": <int>,            # 0 for dry_run
                "deleted_ids": [<id>, ...]   # empty list for dry_run
            }
        """
        try:
            with self.conn.cursor() as cur:
                # Count how many duplicate rows exist (rows that would be removed)
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

                # Delete duplicates, keep the first row per partition (latest created_date, tie-break by highest id)
                # Use a CTE to identify duplicate ids then delete them and RETURNING the deleted ids.
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

            # commit once outside cursor
            self.conn.commit()

            return {
                "duplicates_found": dup_count,
                "deleted": len(deleted_ids),
                "deleted_ids": deleted_ids
            }

        except Exception:
            # On any error, rollback to avoid partial deletes
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
    # Helper utilities
    # -------------------------
    @staticmethod
    def _name_to_token_set(name: str) -> Set[str]:
        """
        Normalize an author name into a set of tokens.
        Example:
            "He Kaiming" -> {"he","kaiming"}
            "kaiming he" -> {"kaiming","he"}
        This is intentionally simple (split on word characters) and lowercases.
        """
        if not name:
            return set()
        tokens = re.findall(r"\w+", name.lower())
        return set(tokens)

    @staticmethod
    def _parse_pg_text_array(arr_val: Any) -> List[str]:
        """
        Robustly parse a Postgres text[] value returned as either:
          - a Python list (most psycopg2 setups)
          - a Postgres-style string like '{"a b","c d"}'
        Returns a list of strings.
        """
        if arr_val is None:
            return []
        # Already a Python list
        if isinstance(arr_val, (list, tuple)):
            return [str(x) for x in arr_val]

        # If it's a string like '{"he kaiming","ren shaoqing"}'
        s = str(arr_val).strip()
        if s.startswith('{') and s.endswith('}'):
            inner = s[1:-1]
            # Use csv to respect quoted elements and commas inside quotes
            # csv expects a file-like; give it the inner string
            reader = csv.reader([inner], delimiter=',', quotechar='"')
            parsed = next(reader, [])
            # strip possible surrounding quotes/spaces
            return [p.strip() for p in parsed if p is not None and p != '']
        # Fallback: split on commas
        return [p.strip() for p in s.split(',') if p.strip()]

    # -------------------------
    # Main search function
    # -------------------------
    def search_paper(
        self,
        normalized_title: str,
        normalized_authors: List[str],
        title_similarity_threshold: float = 0.4,
        max_candidates: int = 20
    ) -> Union[Dict[str, Any], bool]:
        """
        Search for a single best-matching paper.

        Parameters
        ----------
        normalized_title : str
            The normalized title you are querying (e.g. "deep residual learning for im age recognition")
        normalized_authors : List[str]
            List of normalized author strings from the query (e.g. ['kaiming he', 'xiangyu zhang', ...])
        title_similarity_threshold : float
            Minimal trigram similarity between query title and stored normalized_title.
            (0..1 typical). Default 0.4; raise to be stricter.
        max_candidates : int
            How many title-similar rows to fetch for author comparison.

        Returns
        -------
        dict
            The best-matching row (columns as keys), plus extra keys:
              - _title_similarity: float
              - _author_match_count: int
        or
        False
            If no candidate passes the author-matching requirement.
        """
        if not normalized_title:
            return False

        # prepare normalized token sets for query authors
        query_author_token_sets = [self._name_to_token_set(a) for a in normalized_authors if a and a.strip()]
        if len(query_author_token_sets) == 0:
            # No authors to check -> we cannot validate author matches; return False per requirement.
            return False

        # Execute SQL to get top candidates by title similarity
        sql = """
            SELECT
                id, arxiv_id, title, normalized_title, normalized_authors,
                title_and_authors, created_date, oai_header, metadata,
                similarity(normalized_title, %s) AS _title_similarity
            FROM PAPERS
            WHERE similarity(normalized_title, %s) >= %s
            ORDER BY _title_similarity DESC
            LIMIT %s;
        """

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (normalized_title, normalized_title, title_similarity_threshold, max_candidates))
            rows = cur.fetchall()

        if not rows:
            return False

        best_candidate = None
        best_priority = ( -1.0, -1 )  # (title_similarity, author_match_count)
        for row in rows:
            # Parse DB authors into python list
            db_authors_raw = row.get('normalized_authors')
            db_authors_list = self._parse_pg_text_array(db_authors_raw)

            # Normalize DB authors to token sets
            db_author_token_sets = [self._name_to_token_set(a) for a in db_authors_list if a and a.strip()]

            # Compute how many query authors match any DB author (exact token-set equality)
            matched_query_indices = set()
            for q_idx, qset in enumerate(query_author_token_sets):
                for dset in db_author_token_sets:
                    if qset and dset and qset == dset:
                        matched_query_indices.add(q_idx)
                        break

            author_match_count = len(matched_query_indices)

            # Candidate qualifies only if at least one author matched
            if author_match_count == 0:
                continue

            # priority: higher title similarity first, then more author matches
            title_sim = float(row.get('_title_similarity') or 0.0)
            priority = (title_sim, author_match_count)
            if priority > best_priority:
                best_priority = priority
                best_candidate = row

        if not best_candidate:
            # No candidate had an author match
            return False

        # Attach helpful metadata fields and return
        result = dict(best_candidate)  # RealDictRow -> dict
        # Ensure normalized_authors returned as a Python list
        result['normalized_authors'] = self._parse_pg_text_array(result.get('normalized_authors'))
        # Add meta info
        result['_title_similarity'] = float(best_priority[0])
        result['_author_match_count'] = int(best_priority[1])

        return result

# Usage example
if __name__ == "__main__":
    ndjson_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson"
    PG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "arxiv_meta_db_3",
    "user": "postgres",
    "password": "@Q_Fa;ml$f!@94r"
}
    searcher = ArxivMetaSearchDB(PG)
    # searcher.build_db(ndjson_path, dry_run=False)
    # sys.exit()
    # query = "TRANSIMS traffic flow characteristics | Barrett Christopher L., Donnelly Rick, Nagel Kai, Pieck Martin, Stretz Paula"
    # query = "Sequence transduction with recurrent neural networks | A Graves"
    # authors = "Ankur Parikh, Oscar Täckström, Dipanjan Das, Jakob Uszkoreit"
    # query = {'title': 'A decomposable attention model', 'authors_teiled': ['Ankur Parikh', 'Oscar Täckström', 'Dipanjan Das', 'Jakob Uszkoreit']}
    # query = {'title': 'Google’s neural machine translation system: Bridging the gap between human and machine translation', 'authors_teiled': ['Yonghui Wu', 'Mike Schuster', 'Zhifeng Chen', 'V Quoc', 'Mohammad Le', 'Wolfgang Norouzi', 'Maxim Macherey', 'Yuan Krikun', 'Qin Cao', 'Klaus Gao', 'Macherey']}
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

    for query in queries:
        print(searcher.search_paper(query['title'], query['authors_teiled'], title_similarity_threshold=0.5))
        print('-'*100)