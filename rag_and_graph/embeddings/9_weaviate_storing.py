import weaviate
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union, Optional, Tuple
from weaviate.classes.config import Property, DataType, Configure
from tqdm.auto import tqdm
from config import *
import os

# UPDATES NEEDED:
    # 1. Don't save embeddings to jsonl. Directly save them to weaviate
    # 2. 


# --- Logging Setup ---
logging.basicConfig(
    filename=LOG_DIR,
    filemode='a',
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Type Definitions ---
ChunkData = Dict[str, Union[str, float, int]]
DataObject = Dict[str, Any]

class VectorDBBuilder:
    def __init__(self, device: str = "cuda", batch_size: int = 32):
        """
        Initializes the builder.
        
        Args:
            device: 'cuda' for GPU or 'cpu'.
            batch_size: Number of items to process at once for embedding generation.
        """
        self.device = device
        self.batch_size = batch_size
        self.embedding_model = self._get_embedding_model()

    def _get_embedding_model(self) -> SentenceTransformer:
        logger.info(f"Loading embedding model: {MODEL_PATH} on {self.device}...")
        return SentenceTransformer(MODEL_PATH, device=self.device)

    def _get_client(self) -> Optional[weaviate.WeaviateClient]:
        """
        Connects to the local Weaviate instance.
        """
        try:
            client = weaviate.connect_to_local()
            if not client.is_ready():
                logger.error("[ERROR] Weaviate instance is not ready.")
                client.close()
                return None
            return client
        except Exception as e:
            logger.error(f"[ERROR] Connection error: {e}")
            return None

    def create_weaviate_schema(self) -> None:
        """
        Creates Paper and Chunk collections in Weaviate instance if they don't already exist.
        Using Weaviate Python Client v4 syntax.
        """
        logger.info("--- Defining Weaviate Schema ---")
        
        client = self._get_client()
        if not client:
            return

        try:
            # We use the client as a context manager ensures connection is closed
            with client: 
                # --- Paper Collection ---
                if not client.collections.exists("Paper"):
                    client.collections.create(
                        name="Paper",
                        properties=[
                            Property(name="arxiv_id", data_type=DataType.TEXT),
                            Property(name="title", data_type=DataType.TEXT),
                            Property(name="abstract", data_type=DataType.TEXT),
                        ],
                        # We provide vectors ourselves via SentenceTransformer
                        vector_config=Configure.Vectors.self_provided(),
                    )
                    logger.info("[SUCCESS] Created collection: Paper")
                else:
                    logger.info("[INFO] Collection Paper already exists.")

                # --- Chunk Collection ---
                if not client.collections.exists("Chunk"):
                    client.collections.create(
                        name="Chunk",
                        properties=[
                            Property(name="paper_id", data_type=DataType.TEXT),
                            Property(name="chunk_paper_id", data_type=DataType.TEXT),
                            Property(name="section_title", data_type=DataType.TEXT),
                            Property(name="chunk_text", data_type=DataType.TEXT),
                            Property(name="token_len", data_type=DataType.INT),
                            Property(name="checksum", data_type=DataType.TEXT),
                        ],
                        vector_config=Configure.Vectors.self_provided(),
                    )
                    logger.info("[SUCCESS] Created collection: Chunk")
                else:
                    logger.info("[INFO] Collection Chunk already exists.")

        except Exception as e:
            logger.error(f"An error occurred while creating schema: {e}")
            raise

    def _batch_encode(self, texts: List[str]) -> np.ndarray | list:
        """
        Generates embeddings for a batch of texts.
        Returns numpy array
        """
        if not texts:
            return []
        
        # encode returns a numpy array (shape: (len(texts), embedding_dim), dtype: float32)
        embeddings_array = self.embedding_model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        
        return embeddings_array

    def _prepare_for_encoding(self, big_batch: List[Dict]):
        papers_batch: List[Dict] = [] # 1
        chunks_batch: List[List[Dict]] = [] # n chunks for 1 paper
        for obj in big_batch:
            paper_obj = {
                "properties": {
                    "arxiv_id": obj.get('arxiv_id'),
                    "title": obj.get('title'),
                    "abstract": obj.get('abstract')
                }
            }
            papers_batch.append(paper_obj)
            chunks_batch.append(obj.get('chunks', []))
        return papers_batch, chunks_batch
    
    def _embed_papers(self, papers: List[Dict]) -> List[Dict]:
        papers_batch: List[str] = []
        for paper_obj in papers:
            text_for_embedding = f"{paper_obj['properties']['title']} {paper_obj['properties']['abstract']}"
            papers_batch.append(text_for_embedding)

        embeddings = self._batch_encode(papers_batch)
        
        ready_papers_batch: List[Dict] = []
        for i, paper_obj in enumerate(papers):
            paper_obj['vector'] = embeddings[i]
            ready_papers_batch.append(paper_obj)

        return ready_papers_batch

    def _embed_chunks(self, chunks: List[List[Dict]]) -> List[Dict]:
        ready_chunks_batch: List[Dict] = []
        
        for list_of_dicts in chunks:
            chunks_text_batch: List[str] = []
            for chunk_dict in list_of_dicts:
                chunks_text_batch.append(chunk_dict['chunk_text'])

            embeddings = self._batch_encode(chunks_text_batch)

            for i, chunk_dict in enumerate(list_of_dicts):
                chunks_obj: Dict[str, Any] = {
                    "properties": {
                        "chunk_paper_id": chunk_dict['chunk_paper_id'], 
                        "chunk_id": chunk_dict['chunk_id'], 
                        "section_title": chunk_dict['section_title'], 
                        "chunk_text": chunk_dict['chunk_text'], 
                        "token_len": chunk_dict['token_len'], 
                        "checksum": chunk_dict['checksum']
                    },
                    "vector": embeddings[i]
                }
                ready_chunks_batch.append(chunks_obj)
        
        return ready_chunks_batch

    def jsonl_batch_processing(self, show_progress: bool = True, path: str = JSONL_PATH) -> Tuple[int, int]:
        """
        Read a JSONL file and process items in batches. For every batch of size self.batch_size:
        - prepare for encoding
        - embed papers and chunks
        - insert results into DB via self.batch_db_ingestion

        :param show_progress: whether to wrap the file iterator with tqdm
        :param path: path to the input JSONL file
        :returns: tuple (total_papers_inserted, total_chunks_inserted)
        """
        total_papers_inserted: int = 0
        total_chunks_inserted: int = 0
        big_batch: List[Dict[str, Any]] = []
        iterator = None

        try:
            with open(path, 'r', encoding='utf-8') as f_input:
                iterator = f_input
                if show_progress:
                    # wrap the file iterator with tqdm; we'll close it after the loop
                    iterator = tqdm(f_input, desc="Processing Raw File", unit="lines")

                for line in iterator:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON line")
                        continue

                    big_batch.append(obj)

                    if len(big_batch) >= self.batch_size:
                        papers_batch, chunks_batch = self._prepare_for_encoding(big_batch)

                        # Prepare objecet with their embeddings
                        ready_papers_batch = self._embed_papers(papers_batch)
                        ready_chunks_batch = self._embed_chunks(chunks_batch)

                        # Insert into DB
                        self.batch_db_ingestion("Paper", ready_papers_batch)
                        self.batch_db_ingestion("Chunk", ready_chunks_batch)

                        # update totals
                        total_papers_inserted += len(ready_papers_batch)
                        total_chunks_inserted += len(ready_chunks_batch)

                        # clear the batch for reuse (keeps the same list object)
                        big_batch.clear()

                # flush remaining items
                if big_batch:
                    papers_batch, chunks_batch = self._prepare_for_encoding(big_batch)
                    ready_papers_batch = self._embed_papers(papers_batch)
                    ready_chunks_batch = self._embed_chunks(chunks_batch)

                    self.batch_db_ingestion("Paper", ready_papers_batch)
                    self.batch_db_ingestion("Chunk", ready_chunks_batch)

                    total_papers_inserted += len(ready_papers_batch)
                    total_chunks_inserted += len(ready_chunks_batch)

        except FileNotFoundError:
            logger.error(f"Input file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error during preparation: {e}")
            raise
        finally:
            try:
                if show_progress and iterator is not None:
                    iterator.close()
            except Exception:
                # don't mask original exceptions if any; just log the close failure
                logger.debug("Failed to close progress iterator", exc_info=True)

        logger.info(f"Processing and ingestion is complete. Total papers and chunks processed: {total_papers_inserted, total_chunks_inserted}")
        return total_papers_inserted, total_chunks_inserted

    def batch_db_ingestion(self, collection_name: str, data_batch: List[Dict]):
        """
        Inserts data batch to weaviate db
        """
        logger.info(f"--- Batch inserting {collection_name} ---")
        
        client = self._get_client()
        if not client:
            return
        try:
            with client: 
                collection = client.collections.use(collection_name)
                with collection.batch.fixed_size(batch_size=200) as batch:
                    for data_row in data_batch:
                        batch.add_object(
                            properties=data_row['properties'],
                            vector = data_row['vector']
                        )
                        if batch.number_errors > 10:
                            logger.error("Batch import stopped due to excessive errors.")
                            break
                failed_objects = collection.batch.failed_objects
                if failed_objects:
                    logger.info(f"Number of failed imports: {len(failed_objects)}")
                    logger.info(f"First failed object: {failed_objects[0]}")
        except Exception as e:
            logger.error(f"An error occurred while creating schema: {e}")
            raise

    def update_jsonl_with_weaviate_ids(self):
        raise NotImplementedError("Not implemented")

    def build_db(self) -> None:
        """
        Orchestrates the entire pipeline.
        """
        self.create_weaviate_schema()
        self.jsonl_batch_processing(path=JSONL_PATH)


if __name__ == "__main__":
    # Example usage
    try:
        builder = VectorDBBuilder(device="cuda", batch_size=BATCH_SIZE)
        builder.build_db()
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")