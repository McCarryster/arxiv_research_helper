import json
import psycopg2
from psycopg2.extras import execute_batch
from psycopg2.extras import RealDictCursor
import time
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class ArxivMetaSearchDB:
    def __init__(self, PG):
        self.conn = psycopg2.connect(**PG)

    def create_schema(self):
        create_arxiv_paper_query = '''
        CREATE TABLE IF NOT EXISTS PAPERS (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT[],
            oai_header JSONB DEFAULT '{}',
            metadata JSONB DEFAULT '{}'
        );
        '''

        with self.conn.cursor() as cur:
            cur.execute(create_arxiv_paper_query)

        self.conn.commit()

    def create_index(self):
        with self.conn.cursor() as cur:
            # Ensure pg_trgm extension exists (optional for similarity search)
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            
            DDL = """
            ALTER TABLE papers
            ADD COLUMN IF NOT EXISTS title_norm TEXT GENERATED ALWAYS AS (lower(title)) STORED,
            ADD COLUMN IF NOT EXISTS authors_norm TEXT GENERATED ALWAYS AS (lower(array_to_string(authors, '||'))) STORED;

            CREATE INDEX IF NOT EXISTS idx_papers_title_norm ON papers (title_norm);
            CREATE INDEX IF NOT EXISTS idx_papers_title_authors_norm ON papers (title_norm, authors_norm);

            -- optional: GIN index on the array for contains/overlap queries
            CREATE INDEX IF NOT EXISTS idx_papers_authors_gin ON papers USING GIN (authors);
            """

            # Drop trigger if exists to avoid duplicate error
            cur.execute(DDL)

        self.conn.commit()

    def insert_json(self, ndjson_path):
        """
        ndjson has the following structure (example)
        {
            "oai_header": {
                "identifier": "oai:arXiv.org:adap-org/9710003",
                "datestamp": "2005-09-17",
                "sets": [
                    "physics:nlin:AO"
                ]
            },
            "metadata": {
                "id": "adap-org/9710003", # It is an actual arxiv_id
                "created": "1997-10-21",
                "updated": "2009-11-30",
                "authors": {
                    "author": [
                        {
                            "keyname": "Nagel",
                            "forenames": "Kai"
                        },
                        {
                            "keyname": "Stretz",
                            "forenames": "Paula"
                        },
                        {
                            "keyname": "Pieck",
                            "forenames": "Martin"
                        },
                        {
                            "keyname": "Donnelly",
                            "forenames": "Rick"
                        },
                        {
                            "keyname": "Barrett",
                            "forenames": "Christopher L."
                        }
                    ]
                },
                "title": "TRANSIMS traffic flow characteristics",
                "categories": "adap-org nlin.AO",
                "comments": "Paper has 23 pages, 5 figures, 28 diagrams",
                "report-no": "adap-org/9710003",
                "abstract": "Knowledge of fundamental traffic flow characteristics of traffic simulation models is an essential requirement when using these models for the planning, design, and operation of transportation systems. In this paper we discuss the following: a description of how features relevant to traffic flow are currently under implementation in the TRANSIMS microsimulation, a proposition for standardized traffic flow tests for traffic simulation models, and the results of these tests for two different versions of the TRANSIMS microsimulation.",
                "journal-ref": "T. Baeck (Ed.) (1997) Proceedings of the Seventh International Conference on Genetic Algorithms (ICGA'97), Morgan Kauffman, 73-80",
                "msc-class": "52B05, 15A69, 14M25, 55N33",
                "doi": "10.1086/312523",
                "proxy": "ccsd ccsd-00000356",
                "acm-class": "None",
                "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
            }
        }
        
        """
        # Make tqdm here to see the progress
        def get_num_lines(file_path):
            with open(file_path, 'r') as f:
                return sum(1 for _ in f)

        total_lines = get_num_lines(ndjson_path)

        with open(ndjson_path, 'r') as f:
            with self.conn.cursor() as cur:
                for line in tqdm(f, total=total_lines):
                    paper = json.loads(line)
                    title = paper['metadata'].get('title')

                    # authors = paper['metadata'].get('authors', {})
                    # print(authors.get('author'), [])
                    # authors = format_authors(authors)
                    authors = paper['metadata'].get('authors', {})
                    # print(authors)
                    # print('-'*100)
                    # print(authors.get('author', []))
                    authors = format_authors(authors.get('author', []))
                    oai_header = json.dumps(paper.get('oai_header', {}))
                    metadata = json.dumps(paper.get('metadata', {}))
                    insert_query = '''
                    INSERT INTO PAPERS (title, authors, oai_header, metadata)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb)
                    '''
                    cur.execute(insert_query, (title, authors, oai_header, metadata))

        self.conn.commit()


    def search_by_title(self, title):
        sql = "SELECT * FROM papers WHERE title_norm = lower(%s);"
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (title,))
            return cur.fetchall()

    def search_by_title_and_authors(self, title, authors_list): 
        # authors_list must be in the same order and formatting as stored.
        authors_text = "||".join(authors_list)   # same delimiter used in the generated column
        sql = "SELECT * FROM papers WHERE title_norm = lower(%s) AND authors_norm = lower(%s);"
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (title, authors_text))
            return cur.fetchall()

    # Search by title only
    # def search_by_title(self, search_term):
    #     query = """
    #     SELECT * FROM PAPERS
    #     WHERE to_tsvector('english', title) @@ to_tsquery(%s);
    #     """
    #     # Convert space-separated terms to & concatenated form for tsquery
    #     tsquery = ' & '.join(search_term.split())
    #     with self.conn.cursor() as cur:
    #         cur.execute(query, (tsquery,))
    #         return cur.fetchall()

    # def search_by_title_and_authors(self, search_params):
    #     title = search_params.get('title', '')
    #     authors_list = search_params.get('authors_teiled', [])

    #     # Convert title to tsquery format (words joined with &)
    #     title_query = ' & '.join(title.split())

    #     # Flatten authors list into words and join with &
    #     authors_words = []
    #     for author in authors_list:
    #         authors_words.extend(author.split())
    #     authors_query = ' & '.join(authors_words)

    #     # Combine title and authors queries with &
    #     full_query = ' & '.join(filter(None, [title_query, authors_query]))

    #     sql = """
    #     SELECT * FROM PAPERS
    #     WHERE search_vector @@ to_tsquery('english', %s);
    #     """

    #     with self.conn.cursor() as cur:
    #         cur.execute(sql, (full_query,))
    #         results = cur.fetchall()
    #     return results

    # def _format_results(self, results):
    #     # Should return pretty dict with this kind of structure {'arxiv_id': ..., 'title': ..., 'authors': ..., 'abstract': ...}


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
    searcher.create_schema()
    searcher.insert_json(ndjson_path)
    searcher.create_index()

    # Search examples
    query = "Generating text with recurrent neural networks"
    query_dict = {'title': 'Generating text with recurrent neural networks', 'authors_teiled': ['I Sutskever', 'J Martens', 'G Hinton']}
    results = searcher.search_by_title(query)
    results = searcher.search_by_title_and_authors(query_dict['title'], query_dict['authors_teiled'])
    # print("Title search results:", results)
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