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

# --- Type Definitions ---
ChunkData = Dict[str, Union[str, float]]
DataObject = Dict[str, Union[ChunkData, List[float]]]

class VectorStoreBuilder:
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        self.embedding_model = self._get_embedding_model()

        # logger.info(f"Initialized VectorStoreBuilder with model '{embedding_model_name}', "
        #             f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, device={device}")

    def _get_client(self) -> Optional[weaviate.WeaviateClient]:
        """
        Connects to the local Weaviate instance.
        """
        try:
            client: weaviate.WeaviateClient = weaviate.connect_to_local(
                host="localhost",
                port=8080 # default port
            )
            if not client.is_ready():
                print("🛑 Weaviate instance is not ready. Please ensure your local Weaviate is running.")
                client.close()
                return None
            
            print(f"✅ Connected to Weaviate at {WEAVIATE_URL}")
            return client
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None

    def _get_embedding_model(self):
        # logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(MODEL_PATH)
        return self.embedding_model

    def create_collection(self):
        """
        Creates collection in Weaviate instance if it does not already exist.
        """
        try:
            client = self._get_client()
            if client:
                if client.collections.exists(COLLECTION_NAME):
                    print(f"Collection '{COLLECTION_NAME}' already exists.")
                else:
                    client.collections.create(
                        name=COLLECTION_NAME,
                        vector_config=Configure.Vectors.self_provided(),  # No automatic vectorization since we're providing vectors
                    )
                    print(f"Collection '{COLLECTION_NAME}' created successfully.")
                client.close()
            else:
                print("There is no Weaviate connection!")
        except Exception as e:
            print(f"❌ Creation error: {e}")
            return None

    def insert_data(self, data_objects):
        """
        Inserts data into Weaviate db, checks if collection exists before insertion.
        """
        try:
            client = self._get_client()
            if client:
                if not client.collections.exists(COLLECTION_NAME):
                    print(f"Collection '{COLLECTION_NAME}' does not exist. Please create it first.")
                    return None
                collection = client.collections.get(COLLECTION_NAME)
                with collection.batch.fixed_size(batch_size=200) as batch:
                    for obj in data_objects:
                        batch.add_object(properties=obj["properties"], vector=obj["vector"])
                client.close()
                print(f"Imported {len(data_objects)} objects with vectors into the collection '{COLLECTION_NAME}'")
            else:
                print("There is no Weaviate connection!")
        except Exception as e:
            print(f"❌ Insertion error: {e}")
            return None

    def get_data(self, vector, limit):
        """
        Gets the data from Weaviate db, checks if collection exists before insertion.
        """
        try:
            client = self._get_client()
            if client:
                if not client.collections.exists(COLLECTION_NAME):
                    print(f"Collection '{COLLECTION_NAME}' does not exist. Please create it first.")
                    return None
                collection = client.collections.get(COLLECTION_NAME)
                response = collection.query.near_vector(
                    near_vector=vector, 
                    limit=limit
                )
                for obj in response.objects:
                    print(json.dumps(obj.properties, indent=2))  # Inspect the results
                client.close()
            else:
                print("There is no Weaviate connection!")
        except Exception as e:
            print(f"❌ Insertion error: {e}")
            return None

if __name__ == "__main__":
    weav_db = VectorStoreBuilder()
    weav_db.create_collection()

    print('-'*120)

    data_objects = [
        {"properties": {"id": "1111.1111",
                        "title": "The Matrix", "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
                        "genre": "Science Fiction"},
                        "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},

        {"properties": {"id": "2222.2222", 
                        "title": "Spirited Away", 
                        "description": "A young girl becomes trapped in a mysterious world of spirits and must find a way to save her parents and return home.", 
                        "genre": "Animation"},
                        "vector": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},

        {"properties": {"id": "3333.3333",
                        "title": "The Lord of the Rings: The Fellowship of the Ring", 
                        "description": "A meek Hobbit and his companions set out on a perilous journey to destroy a powerful ring and save Middle-earth.",
                        "genre": "Fantasy"},
                        "vector": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    ]
    weav_db.insert_data(data_objects)

    print('-'*120)
    
    vector = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81]
    weav_db.get_data(vector, limit=2)