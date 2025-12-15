import arxiv
from neo4j import GraphDatabase
from typing import Dict, List
from dotenv import load_dotenv
import requests
import os
import re
from PyPDF2 import PdfReader
from psycopg2.extras import execute_values
import json
import openai
import psycopg2

load_dotenv()
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./pdfs")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "text-embedding-3-small")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class ArxivProcessor:
    def paper_to_json(self, result: arxiv.Result) -> dict:
        return {
            "entry_id": result.entry_id.split("/")[-1],
            "updated": result.updated.strftime("%Y-%m-%d"),
            "published": result.published.strftime("%Y-%m-%d"),
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "comment": result.comment,
            "journal_ref": result.journal_ref,
            "doi": result.doi,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "links": [str(link.href) for link in result.links],
            "pdf_url": result.pdf_url,
        }

    def download_pdf(self, result):
        """Download PDF for the main paper only."""
        arxiv_id = result.entry_id.split("/")[-1]
        pdf_path = os.path.join(OUTPUT_DIR, f"{arxiv_id}.pdf")
        if os.path.exists(pdf_path):
            return pdf_path

        try:
            r = requests.get(result.pdf_url, timeout=30)
            r.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            return pdf_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {result.pdf_url}: {e}")
            return None

    def extract_arxiv_references(self, pdf_path) -> List[str]:
        """Extract arXiv IDs and arxiv.org URLs from PDF text."""
        if not os.path.exists(pdf_path):
            return []

        try:
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)

            # matches1 = re.findall(r'arXiv:\d{4}\.\d{4,5}(?:v\d+)?', text)
            matches1 = re.findall(r'arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?', text, flags=re.IGNORECASE)
            matches2 = re.findall(r'https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)', text)
            return list(set(matches1 + matches2))
        except Exception as e:
            print(f"Failed to extract arXiv references: {e}")
            return []

    def clean_id(self, ref: str) -> str:
        """Normalize references into plain arXiv IDs (with optional version)."""
        ref = ref.replace("arXiv:", "").strip()  # remove label and extra spaces
        match = re.search(r'(\d{4}\.\d{4,5})(v\d+)?', ref)
        return match.group(0) if match else None # type: ignore

class ArxivGraphDB:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")

    def add_papers_and_citations(self, papers: List[Dict], citations: List[tuple]):
        """Batch insert papers and citation edges."""
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                # Add all papers
                for paper_json in papers:
                    paper_id = paper_json["entry_id"]
                    tx.run(
                        """
                        MERGE (p:Paper {id: $paper_id})
                        SET p.title = $title,
                            p.doi = $doi,
                            p.published = date($published),
                            p.updated = date($updated),
                            p.summary = $summary,
                            p.url = $url
                        """,
                        paper_id=paper_id,
                        title=paper_json["title"],
                        doi=paper_json.get("doi"),
                        published=paper_json["published"],
                        updated=paper_json["updated"],
                        summary=paper_json.get("summary"),
                        url=paper_json.get("pdf_url")
                    )

                    for author in paper_json.get("authors", []):
                        tx.run(
                            """
                            MERGE (a:Author {name: $author})
                            WITH a
                            MATCH (p:Paper {id: $paper_id})
                            MERGE (a)-[:WROTE]->(p)
                            """,
                            author=author,
                            paper_id=paper_id
                        )

                    for category in paper_json.get("categories", []):
                        tx.run(
                            """
                            MERGE (c:Category {name: $category})
                            WITH c
                            MATCH (p:Paper {id: $paper_id})
                            MERGE (p)-[:IN_CATEGORY]->(c)
                            """,
                            category=category,
                            paper_id=paper_id
                        )

                # Add all citations
                for citing_id, cited_id in citations:
                    tx.run(
                        """
                        MATCH (citing:Paper {id: $citing_id})
                        MATCH (cited:Paper {id: $cited_id})
                        MERGE (citing)-[:CITES]->(cited)
                        """,
                        citing_id=citing_id,
                        cited_id=cited_id
                    )
                tx.commit()

# ===== Example Usage =====
try:
    client = arxiv.Client()
    processor = ArxivProcessor()
    db = ArxivGraphDB(NEO4J_URI, NEO4J_USER, NEO4J_PASS) # type: ignore
    db.create_constraints()

    # Step 1: fetch main paper
    search = arxiv.Search(id_list=["2406.17092v1", "2509.16203v1", "1706.03762", "2409.01193v2"])
    for result in client.results(search):
        print("Main paper:", result.title, f"({result.entry_id})")
        pdf_path = processor.download_pdf(result)
        if not pdf_path:
            continue

        # Make chunk



        # Store paper and embeddings to arxiv_sqldb
        title_abstract = result.title + ". " + result.summary

        main_json = processor.paper_to_json(result)

        # Step 2: extract references
        references = processor.extract_arxiv_references(pdf_path)
        ids = list(set(processor.clean_id(ref) for ref in references if processor.clean_id(ref)))
        ids = [arxiv_id for arxiv_id in ids if arxiv_id != result.entry_id.split("/")[-1]]
        print(ids)                                                                                              # !!!!! CHECK FOR CORRECTNESS !!!!!

        print(f"{len(ids)} referenced papers found")

        # Step 3: batch fetch referenced papers metadata
        ref_papers = []
        batch_size = 10
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            search = arxiv.Search(id_list=batch_ids)
            for ref_result in client.results(search):
                ref_papers.append(processor.paper_to_json(ref_result))

        # Step 4: batch insert main + references + citation edges
        all_papers = [main_json] + ref_papers
        citations = [(result.entry_id.split("/")[-1], ref["entry_id"]) for ref in ref_papers]
        db.add_papers_and_citations(all_papers, citations)

    db.close()

except Exception as e:
    print(f"Error: {e}")







# 1. Ingest: Store title + abstract + metadata + embeddings in DB/graph.
# 2. Cluster (batch): Run k-means/HDBSCAN on embeddings → get concept clusters.
# 3. Extract labels: Use abstracts + KeyBERT/TF-IDF → feed to LLM → label concept nodes.
# 4. Update graph: Create/update concept nodes and connect papers.
# 5. Incremental update: For new papers, assign to nearest concept or create a new one.
# 6. Periodic refresh: Reclustering to refine and evolve concepts over time.

# NODES ARE MADE:
    # Paper
    # Author
    # Category
    # Chunk (WORKING ON IT)