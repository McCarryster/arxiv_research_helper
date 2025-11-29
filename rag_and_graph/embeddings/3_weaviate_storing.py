import weaviate
from weaviate.classes.config import Property, DataType, VectorDistances, Configure
import gc
import os
import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union, Optional
from tqdm.auto import tqdm # User instruction: Always use tqdm for long file iterating
from config import *
import logging
from logger_config import get_logger

# --- Type Definitions ---
ChunkData = Dict[str, Union[str, float]]
DataObject = Dict[str, Union[ChunkData, List[float]]]


logger = get_logger(__name__, "weaviate.log")

class VectorStoreBuilder:
    def __init__(self, device: str = "cpu", show_progress: bool = True) -> None:
        """
        device: "cpu" or "cuda" (or "cuda:0" etc. depending on your environment)
        show_progress: whether to show tqdm progress bars when inserting many items
        """
        self.device: str = device
        self.show_progress: bool = show_progress

        self.client: Optional[weaviate.Client] = self._get_client()
        self.embedding_model: Optional[SentenceTransformer] = None
        if self.client:
            self.embedding_model = self._get_embedding_model()

    def _get_client(self) -> Optional[weaviate.Client]:
        """
        Connects to the local Weaviate instance.
        """
        try:
            client = weaviate.Client(url=WEAVIATE_URL)
            # client.is_ready() returns True/False
            if not client.is_ready():
                logger.error("🛑 Weaviate instance is not ready. Please ensure your local Weaviate is running.")
                return None

            logger.info(f"✅ Connected to Weaviate at {WEAVIATE_URL}")
            return client
        except Exception as e:
            logger.exception(f"❌ Connection error: {e}")
            return None

    def _get_embedding_model(self) -> SentenceTransformer:
        """
        Load the local SentenceTransformer model and move to device.
        """
        try:
            model = SentenceTransformer(MODEL_PATH)
            # move to device (SentenceTransformer exposes .to())
            try:
                model.to(self.device)
            except Exception:
                logger.debug("Could not move model to device with .to(); continuing with default device.")
            logger.info(f"Loaded embedding model from: {MODEL_PATH} (device={self.device})")
            return model
        except Exception as e:
            logger.exception(f"❌ Failed to load embedding model from {MODEL_PATH}: {e}")
            raise

    def create_collection(self) -> bool:
        """
        Creates a class in Weaviate with vectorizer disabled (we provide vectors).
        Returns True if class created or already exists, False on error.
        """
        if not self.client:
            logger.error("No weaviate client available.")
            return False

        try:
            schema = self.client.schema.get()
            existing_classes = schema.get("classes", []) if isinstance(schema, dict) else []

            if any(c.get("class") == COLLECTION_NAME for c in existing_classes):
                logger.info(f"Class '{COLLECTION_NAME}' already exists in schema.")
                return True

            class_definition = {
                "class": COLLECTION_NAME,
                "vectorizer": "none",  # IMPORTANT: we provide vectors ourselves
                "properties": [
                    {"name": "title", "dataType": ["text"]},
                    {"name": "description", "dataType": ["text"]},
                    {"name": "genre", "dataType": ["text"]},
                ],
            }
            self.client.schema.create_class(class_definition)
            logger.info(f"Created class '{COLLECTION_NAME}' in Weaviate.")
            return True
        except Exception as e:
            logger.exception(f"❌ Creation error: {e}")
            return False

    def _compute_vector_for_object(self, properties: Dict[str, Any]) -> List[float]:
        """
        Computes embedding for the object. Uses 'description' + 'title' if available.
        Returns a plain Python list of floats.
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded.")

        # Prefer an explicit 'text' or 'description' field for embedding; fallback to title.
        text_to_embed = ""
        if "text" in properties and properties["text"]:
            text_to_embed = str(properties["text"])
        else:
            title = properties.get("title", "")
            description = properties.get("description", "")
            text_to_embed = f"{title}. {description}".strip()

        if not text_to_embed:
            # empty content -> return zero vector of model dimension
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return [0.0] * dim

        emb: np.ndarray = self.embedding_model.encode(text_to_embed, convert_to_numpy=True)
        # Ensure it's python list (JSON serializable)
        return emb.astype(float).tolist()

    def insert_data(self, data_objects: List[Dict[str, Any]]) -> int:
        """
        Inserts data into Weaviate using the batch API.
        Each item in data_objects should be:
          {"properties": {...}, "vector": [...] }   # if vector provided
        or:
          {"properties": {...}}                      # vectors will be computed using embedding model
        Returns the number of objects successfully submitted (may be less on error).
        """
        if not self.client:
            logger.error("No weaviate client available.")
            return 0

        # Ensure the class exists
        if not self.create_collection():
            logger.error("Collection not created and cannot continue with insertion.")
            return 0

        try:
            self.client.batch.configure(batch_size=BATCH_SIZE)
            iterator = data_objects
            if self.show_progress:
                iterator = tqdm(data_objects, desc="Inserting objects", unit="obj")

            inserted_count = 0
            with self.client.batch as batch:
                for obj in iterator:
                    props = obj.get("properties", {})
                    vec = obj.get("vector")
                    if vec is None:
                        # compute vector using embedding model
                        vec = self._compute_vector_for_object(props)
                    # add object to batch
                    batch.add_data_object(data_object=props, class_name=COLLECTION_NAME, vector=vec)
                    inserted_count += 1

            logger.info(f"Imported {inserted_count} objects with vectors into '{COLLECTION_NAME}'")
            return inserted_count
        except Exception as e:
            logger.exception(f"❌ Insertion error: {e}")
            return 0

    def get_data(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a near-vector query on Weaviate and returns the found objects (list of dicts).
        Each dict contains the object's properties and optionally _additional (distance).
        """
        if not self.client:
            logger.error("No weaviate client available.")
            return []

        try:
            # Build query: request the properties we defined and ask for the _additional distance
            query = (
                self.client.query.get(COLLECTION_NAME, ["title", "description", "genre"])
                .with_near_vector({"vector": vector})
                .with_limit(limit)
                .with_additional(["distance"])
            )
            resp = query.do()
            results: List[Dict[str, Any]] = []

            # Response structure: {'data': {'Get': { COLLECTION_NAME: [ { 'title':..., 'description':..., '_additional': {...}}, ... ]}}}
            hits = []
            if isinstance(resp, dict):
                try:
                    hits = resp["data"]["Get"].get(COLLECTION_NAME, [])
                except Exception:
                    hits = []

            for item in hits:
                properties = {k: v for k, v in item.items() if k != "_additional"}
                additional = item.get("_additional", {})
                result = {"properties": properties, "_additional": additional}
                results.append(result)
                # Print for inspection as you had before
                print(json.dumps(result, indent=2))

            return results
        except Exception as e:
            logger.exception(f"❌ Query error: {e}")
            return []


# -----------------------
# Example usage (no CLI)
# -----------------------
if __name__ == "__main__":
    # Update MODEL_PATH and WEAVIATE_URL at top of file as needed before running.

    builder = VectorStoreBuilder(device="cpu", show_progress=True)

    # Create collection (class) in Weaviate
    builder.create_collection()

    # Example objects: two with precomputed vectors, one where vector will be computed from the model
    data_objects_example = [
        {
            "properties": {
                "title": "The Matrix",
                "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
                "genre": "Science Fiction",
            },
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
        {
            "properties": {
                "title": "Spirited Away",
                "description": "A young girl becomes trapped in a mysterious world of spirits and must find a way to save her parents and return home.",
                "genre": "Animation",
            },
            "vector": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        {
            # vector omitted -> will be computed from the local embedding model using title + description
            "properties": {
                "title": "The Lord of the Rings: The Fellowship of the Ring",
                "description": "A meek Hobbit and his companions set out on a perilous journey to destroy a powerful ring and save Middle-earth.",
                "genre": "Fantasy",
            },
        },
    ]

    inserted = builder.insert_data(data_objects_example)
    logger.info(f"Inserted count: {inserted}")

    # Query with a vector (either a precomputed example or computed with the model)
    query_vector = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81]
    results = builder.get_data(vector=query_vector, limit=2)
    logger.info(f"Query returned {len(results)} items")