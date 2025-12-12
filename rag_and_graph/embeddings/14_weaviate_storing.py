import weaviate
import json
import logging
import sys
import weaviate.util
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional, Tuple, Set
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter, MetadataQuery
from vllm.outputs import EmbeddingRequestOutput
from vllm import LLM # Placeholder import
from config import *

# --- Logging Setup (as before) ---
logging.getLogger().handlers.clear()
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)

# --- VLLMQwenEmbedder (as before) ---
class VLLMQwenEmbedder:
    # ... (VLLMQwenEmbedder class as provided - no changes needed) ...
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

# --- Updated VectorDBBuilder Class ---

class VectorDBBuilder():
    def __init__(self, embedder: VLLMQwenEmbedder, batch_size: int = 16):
        self.embedder: VLLMQwenEmbedder = embedder
        self.batch_size: int = batch_size
        self.client: Optional[weaviate.WeaviateClient] = None
    
    def _connect_client(self) -> weaviate.WeaviateClient:
        """
        Connects to the local Weaviate instance and sets self.client.
        Exits on failure.
        """
        if self.client and self.client.is_ready():
            return self.client
            
        try:
            client = weaviate.connect_to_local()
            if not client.is_ready():
                logger.error("Weaviate instance is not ready.")
                client.close()
                sys.exit(1)
            self.client = client
            return client
        except Exception as e:
            logger.critical(f"Connection error: {e}")
            sys.exit(1)

    def create_weaviate_schema(self) -> None:
        """
        Creates Paper and Chunk collections in Weaviate instance if they don't already exist.
        """
        logger.info("--- Defining Weaviate Schema ---")
        client = self._connect_client()
        
        try:
            # --- Paper Collection ---
            if not client.collections.exists("Paper"):
                client.collections.create(
                    name="Paper",
                    properties=[
                        Property(name="arxiv_id", data_type=DataType.TEXT, index_filterable=True),
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="abstract", data_type=DataType.TEXT),
                    ],
                    vector_config=Configure.Vectors.self_provided(),
                )
                logger.info("[SUCCESS] Created collection: Paper")
            else:
                logger.info("Collection Paper already exists.")

            # --- Chunk Collection ---
            if not client.collections.exists("Chunk"):
                client.collections.create(
                    name="Chunk",
                    properties=[
                        Property(name="paper_id", data_type=DataType.TEXT, index_filterable=True),
                        Property(name="chunk_paper_id", data_type=DataType.TEXT, index_filterable=True),
                        Property(name="section_title", data_type=DataType.TEXT),
                        Property(name="chunk_text", data_type=DataType.TEXT),
                        Property(name="token_len", data_type=DataType.INT),
                        Property(name="checksum", data_type=DataType.TEXT, index_filterable=True),
                    ],
                    vector_config=Configure.Vectors.self_provided(),
                )
                logger.info("[SUCCESS] Created collection: Chunk")
            else:
                logger.info("Collection Chunk already exists.")

        except Exception as e:
            logger.error(f"An error occurred while creating schema: {e}")
            raise

    def _batch_check_exists(self, collection_name: str, identifiers: List[str]) -> Set[str]:
        """
        Performs a batch existence check for a list of identifiers against a Weaviate collection.
        
        Args:
            collection_name: The collection name ("Paper" or "Chunk").
            identifiers: List of identifiers (arxiv_id or checksum) to check.
            
        Returns:
            Set[str]: A set of identifiers that already exist in the collection.
        """
        client = self._connect_client()
        if not identifiers:
            return set()
            
        id_map: Dict[str, str] = {
            "Paper": "arxiv_id",
            "Chunk": "checksum"
        }
        target_property: Optional[str] = id_map.get(collection_name)
        
        if not target_property:
            logger.warning(f"Collection '{collection_name}' is not mapped for existence checks.")
            return set()

        existing_ids: Set[str] = set()
        
        # Weaviate's `is_in` filter is used for efficient batch checking.
        try:
            collection = client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                # filters=Filter.by_property(target_property).is_in(identifiers),
                filters=Filter.by_property(target_property).contains_any(identifiers),
                # Request the identifier property back to identify existing records
                return_properties=[target_property], 
                # We only need the properties, not the vector or full metadata
                return_metadata=MetadataQuery(last_update_time=False, is_consistent=False) 
            )

            for obj in response.objects:
                if obj.properties and target_property in obj.properties:
                    existing_ids.add(str(obj.properties[target_property]))
        except Exception as e:
            logger.error(f"Error during batch existence check on {collection_name}: {e}")
            # In case of an error, we proceed cautiously and assume nothing exists to prevent data loss
            return set()

        return existing_ids

    def get_collection_length(self, collection_name: str) -> int:
        """
        Return the total number of objects in a given Weaviate collection.
        """
        client = self._connect_client()
        try:
            collection = client.collections.get(collection_name)
            total = len(collection)  # Use built-in len() instead
            logger.info(f"Collection '{collection_name}' contains {total} objects.")
            return total
        except Exception as e:
            logger.error(f"Error getting length for collection {collection_name}: {e}")
            return 0

    def _prepare_for_encoding(self, big_batch: List[Dict]) -> Tuple[List[Dict], List[List[Dict]]]:
        """
        Transforms a batch of paper data into two lists: one with paper metadata 
        and another with corresponding content chunks.
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
        Read a JSONL file and process items in batches, deduplicating BEFORE embedding.

        :param show_progress: whether to wrap the file iterator with tqdm
        :param path: path to the input JSONL file
        :returns: tuple (total_papers_inserted, total_chunks_inserted)
        """
        total_papers_inserted: int = 0
        total_chunks_inserted: int = 0
        raw_batch: List[Dict[str, Any]] = []
        
        try:
            with open(path, 'r', encoding='utf-8') as f_input:
                iterator = tqdm(f_input, desc="Processing Raw File", unit="lines", disable=not show_progress)

                for line in iterator:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid JSON line")
                        continue

                    raw_batch.append(obj)

                    if len(raw_batch) >= self.batch_size:
                        # 1. Prepare Paper and Chunk identifiers for batch existence check
                        paper_ids_to_check: List[str] = [p["arxiv_id"] for p in raw_batch if "arxiv_id" in p]
                        # Extract all chunk checksums from all papers in the raw batch
                        chunk_checksums_to_check: List[str] = []
                        for p in raw_batch:
                            if "chunks" in p and isinstance(p["chunks"], list):
                                chunk_checksums_to_check.extend([c["checksum"] for c in p["chunks"] if "checksum" in c])

                        # 2. Query Weaviate in two efficient batch operations
                        existing_paper_ids: Set[str] = self._batch_check_exists("Paper", paper_ids_to_check)
                        existing_chunk_checksums: Set[str] = self._batch_check_exists("Chunk", chunk_checksums_to_check)
                        
                        # 3. Filter the raw batch to get the list of objects that need embedding
                        papers_to_embed: List[Dict[str, Any]] = []
                        chunks_to_embed_list: List[List[Dict[str, Any]]] = []

                        for raw_paper_obj in raw_batch:
                            paper_id = raw_paper_obj["arxiv_id"]
                            
                            # Filter Paper: Only embed if the paper is new
                            if paper_id and paper_id not in existing_paper_ids:
                                # Filter Chunks within the new paper:
                                dedup_chunks: List[Dict[str, Any]] = []
                                if "chunks" in raw_paper_obj and isinstance(raw_paper_obj["chunks"], list):
                                    for chunk in raw_paper_obj["chunks"]:
                                        checksum = chunk["checksum"]
                                        if checksum and checksum in existing_chunk_checksums:
                                            # logger.debug(f"Skipping duplicate Chunk {checksum} in new paper {paper_id}")
                                            logger.info(f"Skipping duplicate Chunk: {chunk['chunk_num'], chunk['chunk_paper_id'], chunk['token_len'], chunk['section_title']}")
                                            continue
                                        dedup_chunks.append(chunk)
                                
                                raw_paper_obj["chunks"] = dedup_chunks
                                papers_to_embed.append(raw_paper_obj)
                                chunks_to_embed_list.append(dedup_chunks)
                            else:
                                logger.info(f"Skipping duplicate Paper {paper_id} and its chunks.")
                        
                        # 4. Embed only the filtered data (papers_to_embed & chunks_to_embed_list)
                        ready_papers_batch = self._embed_papers(papers_to_embed)
                        ready_chunks_batch = self._embed_chunks(chunks_to_embed_list) 

                        # 5. Insert into DB
                        self.batch_db_ingestion("Paper", ready_papers_batch)
                        self.batch_db_ingestion("Chunk", ready_chunks_batch)

                        # update totals
                        total_papers_inserted += len(ready_papers_batch)
                        total_chunks_inserted += len(ready_chunks_batch)

                        # clear the batch for reuse
                        raw_batch.clear()

                # flush remaining items
                if raw_batch:
                    # Repeat steps 1-5 for the remaining batch
                    paper_ids_to_check: List[str] = [p["arxiv_id"] for p in raw_batch if "arxiv_id" in p]
                    chunk_checksums_to_check: List[str] = []
                    for p in raw_batch:
                        if "chunks" in p and isinstance(p["chunks"], list):
                            chunk_checksums_to_check.extend([c["checksum"] for c in p["chunks"] if "checksum" in c])

                    existing_paper_ids: Set[str] = self._batch_check_exists("Paper", paper_ids_to_check)
                    existing_chunk_checksums: Set[str] = self._batch_check_exists("Chunk", chunk_checksums_to_check)
                    
                    papers_to_embed: List[Dict[str, Any]] = []
                    chunks_to_embed_list: List[List[Dict[str, Any]]] = []

                    for raw_paper_obj in raw_batch:
                        paper_id = raw_paper_obj.get("arxiv_id")
                        
                        if paper_id and paper_id not in existing_paper_ids:
                            dedup_chunks: List[Dict[str, Any]] = []
                            if "chunks" in raw_paper_obj and isinstance(raw_paper_obj["chunks"], list):
                                for chunk in raw_paper_obj["chunks"]:
                                    checksum = chunk.get("checksum")
                                    if checksum and checksum in existing_chunk_checksums:
                                        continue
                                    dedup_chunks.append(chunk)
                            
                            raw_paper_obj["chunks"] = dedup_chunks
                            papers_to_embed.append(raw_paper_obj)
                            chunks_to_embed_list.append(dedup_chunks)

                    ready_papers_batch = self._embed_papers(papers_to_embed)
                    ready_chunks_batch = self._embed_chunks(chunks_to_embed_list) 

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

        logger.info(f"Processing and ingestion is complete. Total papers and chunks processed: {total_papers_inserted, total_chunks_inserted}")
        return total_papers_inserted, total_chunks_inserted

    def batch_db_ingestion(self, collection_name: str, data_batch: List[Dict[str, Any]]):
        """
        Inserts data batch to weaviate db.
        """
        logger.info(f"--- Batch inserting {collection_name} ({len(data_batch)} objects) ---")
        client = self._connect_client()
        
        try:
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
                logger.info(f"Number of failed imports for {collection_name}: {len(failed_objects)}")
                if len(failed_objects) > 0:
                    logger.info(f"First failed object (example): {failed_objects[0]}")
        except Exception as e:
            logger.error(f"An error occurred during batch ingestion for {collection_name}: {e}")
            raise

    def build_db(self) -> None:
        """
        Orchestrates the entire pipeline.
        """
        try:
            self.create_weaviate_schema()
            self.jsonl_batch_processing(show_progress=True, path=JSONL_PATH)
        finally:
            if self.client:
                self.client.close()
                
    def search_paper_weaviate_by_id(self, collection_name: str, arxiv_id: str) -> Optional[Dict]:
        client = self._connect_client()
        
        try:
            collection = client.collections.use(collection_name)
            response = collection.query.fetch_objects(
                filters=Filter.by_property("arxiv_id").equal(arxiv_id),
                include_vector=True,
                limit=1 
            )

            if response.objects:
                o = response.objects[0]
                logger.info(f"Search found paper with UUID: {o.uuid}")
                return {
                    "uuid": o.uuid,
                    "properties": o.properties,
                    "vector": o.vector
                }
            else:
                logger.info(f"Paper with ID {arxiv_id} not found in {collection_name}.")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for paper {arxiv_id}: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    try:
        embedder = VLLMQwenEmbedder(batch_size=BATCH_SIZE, gpu_memory_utilization=GPU_MEM_UTIL)
        builder = VectorDBBuilder(embedder=embedder, batch_size=BATCH_SIZE)
        print(builder.get_collection_length("Paper"))        
        # builder.build_db() 
        
        # # Example Search
        # builder = VectorDBBuilder(embedder=embedder, batch_size=BATCH_SIZE)
        # result = builder.search_paper_weaviate_by_id("Paper", "1706.03762")
        # if result:
        #     print(f"Search Result for 1706.03762:\n{json.dumps(result, indent=4)}")
        
        if builder.client:
            builder.client.close()
            
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")