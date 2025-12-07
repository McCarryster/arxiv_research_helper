import weaviate
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union, Optional, Iterable, Generator
from weaviate.classes.config import Property, DataType, Configure
from tqdm.auto import tqdm
from config import *

# --- Logging Setup ---
logging.basicConfig(
    filename=log_dir,
    filemode='a',
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Type Definitions ---
ChunkData = Dict[str, Union[str, float, int]]
DataObject = Dict[str, Any]

class VectorStoreBuilder:
    def __init__(self, device: str = "cuda", batch_size: int = 64):
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
                logger.error("🛑 Weaviate instance is not ready.")
                client.close()
                return None
            return client
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
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
                    logger.info("✅ Created collection: Paper")
                else:
                    logger.info("ℹ️ Collection Paper already exists.")

                # --- Chunk Collection ---
                if not client.collections.exists("Chunk"):
                    client.collections.create(
                        name="Chunk",
                        properties=[
                            Property(name="paper_id", data_type=DataType.TEXT),
                            Property(name="chunk_id", data_type=DataType.UUID),
                            Property(name="section_title", data_type=DataType.TEXT),
                            Property(name="chunk_text", data_type=DataType.TEXT),
                            Property(name="token_len", data_type=DataType.INT),
                            Property(name="checksum", data_type=DataType.TEXT),
                        ],
                        vector_config=Configure.Vectors.self_provided(),
                    )
                    logger.info("✅ Created collection: Chunk")
                else:
                    logger.info("ℹ️ Collection Chunk already exists.")

        except Exception as e:
            logger.error(f"An error occurred while creating schema: {e}")
            raise

    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.
        Returns a list of lists (JSON serializable), not numpy arrays.
        """
        if not texts:
            return []
        
        # encode returns a numpy array
        embeddings_array = self.embedding_model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        
        # Convert numpy array to standard list for JSON serialization
        return embeddings_array.tolist()

    def prepare_papers_for_weaviate(self, show_progress: bool = True) -> None:
        """
        Reads raw JSONL, generates embeddings in batches (for speed), 
        and writes prepared JSONL files for ingestion.
        """
        logger.info("--- Preparing Data & Generating Embeddings ---")

        # Buffers to hold data for batch processing
        paper_batch: List[Dict[str, Any]] = []
        chunk_batch: List[Dict[str, Any]] = []
        
        # We need to map which text corresponds to which object in the batch
        paper_texts: List[str] = []
        chunk_texts: List[str] = []

        try:
            with open(JSONL_PATH, 'r', encoding='utf-8') as f_input, \
                 open(WEAVIATE_READY_PAPERS, 'w', encoding='utf-8') as f_papers_out, \
                 open(WEAVIATE_READY_CHUNKS, 'w', encoding='utf-8') as f_chunks_out:

                iterator = f_input
                if show_progress:
                    iterator = tqdm(f_input, desc="Processing Raw File", unit="lines")

                for line in iterator:
                    if not line.strip():
                        continue
                    
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON line")
                        continue

                    # --- 1. Accumulate Paper Data ---
                    text_for_embedding = f"{obj.get('title', '')} {obj.get('abstract', '')}"
                    
                    paper_obj = {
                        "properties": {
                            "arxiv_id": obj.get('arxiv_id'),
                            "title": obj.get('title'),
                            "abstract": obj.get('abstract')
                        }
                    }
                    paper_batch.append(paper_obj)
                    paper_texts.append(text_for_embedding)

                    # --- 2. Accumulate Chunk Data ---
                    for chunk in obj.get('chunks', []):
                        chunk_obj = {
                            "properties": {
                                "chunk_paper_id": chunk.get('chunk_paper_id'),
                                "chunk_id": chunk.get('chunk_id'),
                                "section_title": chunk.get('section_title'),
                                "chunk_text": chunk.get('chunk_text'),
                                "token_len": chunk.get('token_len'),
                                "checksum": chunk.get('checksum')
                            }
                        }
                        chunk_batch.append(chunk_obj)
                        chunk_texts.append(chunk.get('chunk_text', ''))

                    # --- 3. Process Batches if Full ---
                    # To keep memory usage low, we process and write to disk frequently
                    if len(paper_batch) >= self.batch_size:
                        self._process_and_write_batch(paper_batch, paper_texts, f_papers_out)
                        paper_batch = []
                        paper_texts = []

                    if len(chunk_batch) >= self.batch_size:
                        self._process_and_write_batch(chunk_batch, chunk_texts, f_chunks_out)
                        chunk_batch = []
                        chunk_texts = []

                # --- 4. Process Remaining Data ---
                if paper_batch:
                    self._process_and_write_batch(paper_batch, paper_texts, f_papers_out)
                if chunk_batch:
                    self._process_and_write_batch(chunk_batch, chunk_texts, f_chunks_out)

            logger.info("✅ Data preparation complete.")

        except FileNotFoundError:
            logger.error(f"Input file not found: {JSONL_PATH}")
            raise
        except Exception as e:
            logger.error(f"Error during preparation: {e}")
            raise

    def _process_and_write_batch(self, 
                                 data_objects: List[Dict[str, Any]], 
                                 texts: List[str], 
                                 file_handle) -> None:
        """
        Helper to run embeddings on a batch and write to file.
        """
        if not data_objects:
            return

        # 1. Vectorize the texts in one go (Industry Standard optimization)
        vectors = self.batch_encode(texts)

        # 2. Attach vectors to objects and write
        for obj, vector in zip(data_objects, vectors):
            obj["vector"] = vector # vector is now a List[float], valid for JSON
            file_handle.write(json.dumps(obj) + '\n')

    def ingest_from_jsonl(self, file_path: str, collection_name: str, show_progress: bool = True) -> None:
        """
        Reads a prepared JSONL file (with vectors) and batch inserts into Weaviate.
        """
        
        def _file_generator() -> Generator[Dict[str, Any], None, None]:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Count lines for tqdm total if needed, or just iterate
                iterator = f
                if show_progress:
                     iterator = tqdm(f, desc=f"Ingesting {collection_name}", unit="objs")
                
                for line in iterator:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

        logger.info(f"Starting ingestion for collection: {collection_name} from {file_path}")
        
        client = self._get_client()
        if not client:
            return

        try:
            with client:
                collection = client.collections.get(collection_name)
                
                # Weaviate v4 Batch Context Manager
                # This handles buffering and background sending automatically
                with collection.batch.fixed_size(batch_size=100) as batch:
                    for data in _file_generator():
                        batch.add_object(
                            properties=data["properties"],
                            vector=data["vector"]
                        )
                        
                        if batch.number_errors > 10:
                            logger.error("🛑 Too many batch errors. Stopping.")
                            break
                
                # Check for errors after the batch process finishes
                if len(collection.batch.failed_objects) > 0:
                    logger.error(f"Failed to import {len(collection.batch.failed_objects)} objects.")
                    for failed in collection.batch.failed_objects[:3]:
                         logger.error(f"Sample error: {failed.message}")
                else:
                    logger.info(f"✅ Ingestion complete for {collection_name}. No errors reported.")

        except Exception as e:
            logger.error(f"An error occurred during ingestion: {e}")
            raise

    def build_db(self) -> None:
        """
        Orchestrates the entire pipeline.
        """
        self.create_weaviate_schema()
        self.prepare_papers_for_weaviate(show_progress=True)
        self.ingest_from_jsonl(WEAVIATE_READY_PAPERS, 'Paper', show_progress=True)
        self.ingest_from_jsonl(WEAVIATE_READY_CHUNKS, 'Chunk', show_progress=True)

if __name__ == "__main__":
    # Example usage
    try:
        builder = VectorStoreBuilder(device="cuda", batch_size=8) # Change to "cpu" if needed
        builder.build_db()
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")