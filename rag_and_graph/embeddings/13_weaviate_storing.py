import weaviate
import json
import logging
from vllm.outputs import EmbeddingRequestOutput
from vllm import LLM
from typing import List, Dict, Any, Optional, Tuple
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter
import weaviate.util
from tqdm.auto import tqdm
from config import *
import sys

# UPDATES NEEDED:
    # 1. Remake embeddings 16bit (bf16) from 32bit - V
    # 2. Make duplicate insert prevention logic - 
    # 3. Write function that searches jsonl and weaviate db and return full ready object


# --- Logging Setup ---
logging.getLogger().handlers.clear()  # Remove default StreamHandler
logging.basicConfig(
    filename=LOG_DIR,
    filemode='a',
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    force=True  # Python 3.8+: Forces reconfiguration
)
logger = logging.getLogger(__name__)


class VLLMQwenEmbedder:
    def __init__(self, device: str = "cuda", batch_size: int = 16, gpu_memory_utilization:float = 0.7):
        """
        Initializes the VLLM engine with hf model.
        
        Args:
            device: 'cuda' for GPU or 'cpu'.
            batch_size: Number of items to process at once for embedding generation.
        """
        self.device = device
        self.batch_size = batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.embedding_model = self._get_embedding_model()

    def _get_embedding_model(self) -> LLM:
            """
            Loads the vLLM engine.
            """
            logger.info(f"Loading vLLM embedding model: {MODEL_PATH}...")
            
            # vLLM defaults to CUDA.
            return LLM(
                model=MODEL_PATH,
                enforce_eager=True,      # is often safer for embedding tasks to avoid graph capture issues on irregular sizes
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype="bfloat16",        # use bfloat16 for faster computation
                seed=9
            )

    def batch_encode(self, texts: List[str]) -> List[EmbeddingRequestOutput]:
        """
        Generates embeddings for a batch of texts using vLLM.
        
        Args:
            texts: List of strings to encode.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if not texts:
            return []

        outputs = self.embedding_model.embed(texts, use_tqdm=False)
        return outputs

class VectorDBBuilder(VLLMQwenEmbedder):
    def __init__(self, embedder: VLLMQwenEmbedder, batch_size: int = 16):
        self.embedder = embedder
        self.batch_size = batch_size
    
    def _get_client(self) -> Optional[weaviate.WeaviateClient]:
        """
        Connects to the local Weaviate instance or exits on failure.
        """
        try:
            client = weaviate.connect_to_local()
            if not client.is_ready():
                logger.error("Weaviate instance is not ready.")
                client.close()
                sys.exit(1)  # Use exit code for better debugging
            return client
        except Exception as e:
            logger.error(f"Connection error: {e}")
            sys.exit(1)

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
                            Property(name="arxiv_id", data_type=DataType.TEXT, index_filterable=True),
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
                            Property(name="paper_id", data_type=DataType.TEXT, index_filterable=True),
                            Property(name="chunk_paper_id", data_type=DataType.TEXT),
                            Property(name="section_title", data_type=DataType.TEXT),
                            Property(name="chunk_text", data_type=DataType.TEXT),
                            Property(name="token_len", data_type=DataType.INT),
                            Property(name="checksum", data_type=DataType.TEXT, index_filterable=True),
                        ],
                        vector_config=Configure.Vectors.self_provided(),
                    )
                    logger.info("[SUCCESS] Created collection: Chunk")
                else:
                    logger.info("[INFO] Collection Chunk already exists.")

        except Exception as e:
            logger.error(f"An error occurred while creating schema: {e}")
            raise

    def _check_exists(self, collection_name: str, identifier_value: str) -> bool | None:
            """
            Checks if a record exists in a specific collection based on a primary identifier.
            
            Args:
                collection_name: The name of the collection ("Paper" or "Chunk").
                identifier_value: The value to look for (arxiv_id for Paper, checksum for Chunk).
                show_progress: Whether to show a tqdm progress bar (default False for inner loops).
                
            Returns:
                bool: True if the record exists, False otherwise.
            """
            
            client = self._get_client()
            if not client:
                return
            
            # Mapping collection names to their primary filterable property
            id_map: Dict[str, str] = {
                "Paper": "arxiv_id",
                "Chunk": "checksum"
            }

            try:
                with client: 
                    target_property: Optional[str] = id_map.get(collection_name)
                    if not target_property:
                        logger.warning(f"Collection '{collection_name}' is not mapped for existence checks.")
                        return False
                    collection = client.collections.get(collection_name)
                    # Fetch only 1 object with no properties to verify existence efficiently
                    response = collection.query.fetch_objects(
                        filters=Filter.by_property(target_property).equal(identifier_value),
                        limit=1,
                        return_properties=[] # We don't need the data, just the count/existence
                    )
                    return len(response.objects) > 0
            except Exception as e:
                logger.error(f"Error checking existence in {collection_name}: {e}")
                return False

    def _prepare_for_encoding(self, big_batch: List[Dict]) -> Tuple[List[Dict], List[List[Dict]]]:
        """
        Transforms a batch of paper data into two lists: one with paper metadata 
        (arXiv ID, title, abstract) and another with corresponding content chunks.

        Args:
            big_batch (List[Dict]): A list of paper dictionaries, each containing
                metadata and optionally a list of text chunks under 'chunks'.

        Returns:
            Tuple[List[Dict], List[List[Dict]]]: 
                - papers_batch: List of dicts with extracted paper properties.
                - chunks_batch: List of lists, each containing chunk dictionaries for a paper.
        """
        papers_batch: List[Dict] = []
        chunks_batch: List[List[Dict]] = []
        for obj in big_batch:
            papers_batch.append(obj)
            chunks_batch.append(obj.get('chunks', []))
        return papers_batch, chunks_batch

    def _embed_papers(self, papers: List[Dict[str, Any]]) -> List[Dict]:
        papers_batch: List[str] = []
        for paper_obj in papers:
            text_for_embedding = f"{paper_obj['title']} {paper_obj['abstract']}"
            papers_batch.append(text_for_embedding)

        outputs = self.embedder.batch_encode(papers_batch)

        ready_papers_batch: List[Dict[str, Any]] = []
        for paper_obj, output in zip(papers, outputs):
            embeds = output.outputs.embedding
            paper_uuid = weaviate.util.generate_uuid5(paper_obj['arxiv_id'])

            paper_obj: Dict[str, Any] = {
                "properties": {
                    "arxiv_id": paper_obj['arxiv_id'],
                    "title": paper_obj['title'],
                    "abstract": paper_obj['abstract']
                },
                "vector": embeds,
                "uuid": paper_uuid
            }
            ready_papers_batch.append(paper_obj)

        return ready_papers_batch

    def _embed_chunks(self, chunks: List[List[Dict]]) -> List[Dict]:
        ready_chunks_batch: List[Dict] = []
        
        for list_of_dicts in chunks:
            chunks_text_batch: List[str] = []
            for chunk_dict in list_of_dicts:
                chunks_text_batch.append(chunk_dict['chunk_text'])

            outputs = self.embedder.batch_encode(chunks_text_batch)

            for chunk_dict, output in zip(list_of_dicts, outputs):
                embeds = output.outputs.embedding
                chunk_uuid = weaviate.util.generate_uuid5(chunk_dict['checksum'])
                chunks_obj: Dict[str, Any] = {
                    "properties": {
                        "chunk_paper_id": chunk_dict['chunk_paper_id'], 
                        "chunk_id": chunk_dict['chunk_id'], 
                        "section_title": chunk_dict['section_title'], 
                        "chunk_text": chunk_dict['chunk_text'], 
                        "token_len": chunk_dict['token_len'], 
                        "checksum": chunk_dict['checksum']
                    },
                    "vector": embeds,
                    "uuid": chunk_uuid
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

                    # 1. Paper Level Check. If the Paper exists, we assume the whole object is a duplicate and skip it entirely.
                    arxiv_id = obj.get("arxiv_id")
                    if arxiv_id and self._check_exists("Paper", arxiv_id):
                        logger.info(f"Skipping duplicate Paper: {arxiv_id}")
                        continue

                    # 2. Chunk Level Check. If Paper is new, we still need to check if specific chunks inside it already exist.
                    if "chunks" in obj and isinstance(obj["chunks"], list):
                        dedup_chunks: List[Dict[str, Any]] = []
                        for chunk in obj["chunks"]:
                            checksum = chunk["checksum"]
                            if checksum and self._check_exists("Chunk", checksum):
                                logger.info(f"Skipping duplicate Chunk: {obj['chunk_num'], obj['chunk_paper_id'], obj['token_len'], obj['section_title']}")
                                continue
                            
                            dedup_chunks.append(chunk)
                        
                        # Replace the original list with the deduplicated list
                        obj["chunks"] = dedup_chunks

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
                            vector = data_row['vector'],
                            uuid=data_row['uuid']
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

    def build_db(self) -> None:
        """
        Orchestrates the entire pipeline.
        """
        self.create_weaviate_schema()
        self.jsonl_batch_processing(path=JSONL_PATH)

    def search_paper_weaviate_by_id(self, collection_name: str, arxiv_id: str) -> Optional[Dict]:
        client = self._get_client()
        if not client:
            return None
        
        try:
            with client:
                collection = client.collections.use(collection_name)
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("arxiv_id").equal(arxiv_id),
                    include_vector=True,
                    limit=3
                )

                for o in response.objects:
                    print(f"UUID: {o.uuid}")  # Weaviate-generated UUID
                    print(f"Vector: {o.vector}")
                    print(f"Properties: {o.properties}")
                    return {
                        "uuid": o.uuid,  # Weaviate-generated UUID
                        "properties": o.properties,
                        "vector": o.vector
                    }
                    
        except Exception as e:
            logger.error(f"Error searching for paper {arxiv_id}: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    try:
        embedder = VLLMQwenEmbedder(batch_size=BATCH_SIZE, gpu_memory_utilization=0.7)
        builder = VectorDBBuilder(embedder=embedder)
        builder.build_db()
        builder.search_paper_weaviate_by_id("Paper", "1706.03762")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")