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
        self.vectorstore = None

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
            else:
                print("There is no Weaviate connection!")
        except Exception as e:
            print(f"❌ Creation error: {e}")
            return None


    def insert_data(self, data_objects):
        """
        Inserts data into Weaviate db.
        """
        try:
            if self.client:
                collection = self.client.collections.get(COLLECTION_NAME)
                with collection.batch.fixed_size(batch_size=200) as batch:
                    for obj in data_objects:
                        batch.add_object(properties=obj["properties"], vector=obj["vector"])
                self.client.close()
                print(f"Imported {len(data_objects)} objects with vectors into the Movie collection")
            else:
                print(f"There is no weaviate connection!")
        except Exception as e:
            print(f"❌ Insertion error: {e}")
            return None

    def get_data(self, vector, limit):
        """
        Gets data from Weaviate db.
        """
        try:
            if self.client:
                collection = self.client.collections.get(COLLECTION_NAME)
                response = collection.query.near_vector(
                    near_vector=vector, 
                    limit=limit
                )
                for obj in response.objects:
                    print(json.dumps(obj.properties, indent=2))  # Inspect the results
                self.client.close()
            else:
                print(f"There is no weaviate connection!")
        except Exception as e:
            print(f"❌ Insertion error: {e}")
            return None

    # def run_schema_definition(self, show_progress: bool = True) -> None:
    #     """
    #     Main function to connect to Weaviate and define the schema.
    #     """
    #     print(f"Connecting to Weaviate at {WEAVIATE_URL}...")
    #     try:
    #         # Connect to your local Weaviate instance
    #         client: weaviate.WeaviateClient = weaviate.connect_to_local(
    #             host="localhost",
    #             port=8080 # default port
    #         )
            
    #         # Check if the client is connected
    #         if not client.is_ready():
    #             print("🛑 Weaviate instance is not ready. Please ensure your local Weaviate is running.")
    #             return

    #         # Clean up any existing collection with the same name for a fresh start
    #         if client.collections.exists(COLLECTION_NAME):
    #             if show_progress: print(f"Removing existing collection: {COLLECTION_NAME}")
    #             client.collections.delete(COLLECTION_NAME)

    #         define_and_create_schema(client)

    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #     finally:
    #         if 'client' in locals() and client.is_connected():
    #             client.close()
    #             print("Connection closed.")


    # def load_vectorstore(self, path: str):
    #     index_file = os.path.join(path, "index.faiss")
    #     pkl_file = os.path.join(path, "index.pkl")

    #     if os.path.exists(index_file) and os.path.exists(pkl_file):
    #         # logger.info(f"Loading existing vectorstore from: {path}")
    #         self.vectorstore = FAISS.load_local(
    #             path,
    #             self._get_embedding_model(),
    #             allow_dangerous_deserialization=True
    #         )
    #         # logger.info("Vectorstore loaded successfully.")
    #     else:
    #         # logger.info(f"No existing vectorstore found at: {path}")
    #         self.vectorstore = None

    # def add_docs(self, docs: List[Document]):
    #     # logger.info(f"Adding {len(docs)} new documents...")
    #     splitter = self._get_splitter()
    #     split_docs = splitter.split_documents(docs)
    #     if self.vectorstore is None:
    #         # logger.info("Vectorstore not loaded — creating new one.")
    #         self.vectorstore = FAISS.from_documents(split_docs, self._get_embedding_model())
    #     else:
    #         self.vectorstore.add_documents(split_docs)
    #     # logger.info(f"Added {len(split_docs)} document chunks to vectorstore.")

    # def get_existing_uids(self) -> set:
    #     existing_uids = set()
    #     if self.vectorstore and self.vectorstore.docstore:
    #         for doc_id, doc in self.vectorstore.docstore._dict.items():
    #             if doc and doc.metadata:
    #                 uid = doc.metadata.get("uid")
    #                 if uid is not None:
    #                     existing_uids.add(uid)
    #     # logger.info(f"Found {len(existing_uids)} existing document UIDs in vectorstore.")
    #     return existing_uids

    # def filter_new_documents(self, cleaned_data: List[dict]) -> List[Document]:
    #     # logger.info("Filtering new documents based on UIDs...")
    #     existing_uids = self.get_existing_uids()
    #     new_docs = [
    #         Document(
    #             page_content=item["text"],
    #             metadata={"uid": item["uid"], "ru_wiki_pageid": item.get("ru_wiki_pageid")}
    #         )
    #         for item in cleaned_data
    #         if item["uid"] not in existing_uids
    #     ]
    #     # logger.info(f"Identified {len(new_docs)} new documents to be indexed.")
    #     return new_docs

    # def save_vectorstore(self, path: str):
    #     if self.vectorstore:
    #         # logger.info(f"Saving vectorstore to: {path}")
    #         self.vectorstore.save_local(path)
    #         # logger.info("Vectorstore saved successfully.")
    #     else:
    #         # logger.info("No vectorstore to save.")

    # def cleanup(self):
    #     # logger.info("Cleaning up resources...")
    #     self.vectorstore = None
    #     self.embedding_model = None
    #     self.splitter = None
    #     gc.collect()
    #     if self.device != "cpu":
    #         torch.cuda.empty_cache()
    #     # logger.info("Cleanup complete.")


if __name__ == "__main__":
    weav_db = VectorStoreBuilder()
    # weav_db.create_collection()

    # data_objects = [
    #     {"properties": {"title": "The Matrix", "description": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.", "genre": "Science Fiction"},
    #     "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
    #     {"properties": {"title": "Spirited Away", "description": "A young girl becomes trapped in a mysterious world of spirits and must find a way to save her parents and return home.", "genre": "Animation"},
    #     "vector": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
    #     {"properties": {"title": "The Lord of the Rings: The Fellowship of the Ring", "description": "A meek Hobbit and his companions set out on a perilous journey to destroy a powerful ring and save Middle-earth.", "genre": "Fantasy"},
    #     "vector": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    # ]
    # weav_db.insert_data(data_objects)

    vector = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81]
    weav_db.get_data(vector, limit=2)