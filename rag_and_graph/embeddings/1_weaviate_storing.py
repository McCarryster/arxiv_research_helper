import weaviate
from weaviate.classes.config import Property, DataType, VectorDistances, Configure
from weaviate.classes.tenants import Tenant
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
from tqdm.auto import tqdm # User instruction: Always use tqdm for long file iterating
from config import *

# --- Type Definitions ---
ChunkData = Dict[str, Union[str, float]]
DataObject = Dict[str, Union[ChunkData, List[float]]]

class WeaviateIngestor:
    """
    Handles connection, schema creation, vector generation, and data ingestion
    for a Weaviate instance using a custom local embedding model (BYOV).
    """

    def __init__(self, url: str, model_id: str, collection_name: str, show_progress: bool):
        self.url = url
        self.collection_name = collection_name
        self.show_progress = show_progress
        self.client: weaviate.WeaviateClient = weaviate.connect_to_local(host="localhost", port=8080)
        
        # Initialize the local Qwen3 model and tokenizer
        print(f"Loading tokenizer and model: {model_id}...")
        self.model = SentenceTransformer(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Model loaded successfully on {self.device}.")

    def __del__(self) -> None:
        """Closes the Weaviate client connection."""
        if hasattr(self, 'client') and self.client.is_connected():
            self.client.close()
            print("Weaviate client connection closed.")

    def create_schema(self) -> None:
        """
        Defines the Weaviate Collection (Schema).
        Crucially sets 'vectorizer' to 'none' and enables keyword (full-text) search.
        """
        if self.client.collections.exists(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists. Deleting it...")
            self.client.collections.delete(self.collection_name)
        
        print(f"Creating collection '{self.collection_name}'...")
        
        # Collection properties
        properties = [
            weaviate.classes.Property(name="document_id", data_type=weaviate.classes.DataType.TEXT, description="ID of the parent document."),
            weaviate.classes.Property(name="chunk_text", data_type=weaviate.classes.DataType.TEXT, description="The text content to be embedded.", index_filterable=True, index_searchable=True),
            weaviate.classes.Property(name="source_url", data_type=weaviate.classes.DataType.TEXT, description="Source metadata for filtering."),
            weaviate.classes.Property(name="chunk_index", data_type=weaviate.classes.DataType.INT, description="The order index of the chunk.")
        ]
        
        # Configure the collection
        self.client.collections.create(
            name=self.collection_name,
            properties=properties,
            # Set vectorizer to 'none' for BYOV
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
            # Enable full-text/exact search on the text property
            # index_config=weaviate.classes.config.Configure.vec(),
            # Crucially enables the keyword search index (BM25) on text properties
            inverted_index_config=weaviate.classes.config.Configure.inverted_index(
                index_property_length=True # Allows for keyword search on TEXT fields
            )
        )
        print("Schema created successfully with vectorizer='none' and inverted index enabled.")
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates vector embeddings for a list of texts using the Qwen3 model.
        """
        # Tokenize the input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            # Get the model output (last hidden state)
            outputs = self.model(**inputs)
            # Use the CLS token embedding (outputs.last_hidden_state[:, 0]) 
            # or mean pooling, depending on the Qwen model's convention. 
            # Qwen models typically use the first token (CLS) or a specific pooling. 
            # We'll use the first token's output as a common practice for some models.
            embeddings: torch.Tensor = outputs.last_hidden_state[:, 0]
        
        # Normalize embeddings for better performance in vector search
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def ingest_data(self, data: List[ChunkData], batch_size: int = 100) -> None:
        """
        Generates embeddings for all data chunks and imports them in batches.
        """
        print(f"\nStarting data ingestion for {len(data)} chunks...")
        
        chunk_texts: List[str] = [item["chunk_text"] for item in data]
        total_chunks = len(chunk_texts)
        
        # Process and ingest in batches for efficiency
        for i in tqdm(range(0, total_chunks, batch_size), desc="Batch Ingestion Progress", disable=not self.show_progress):
            batch_texts = chunk_texts[i:i + batch_size]
            batch_data = data[i:i + batch_size]
            
            # 1. Generate embeddings for the current batch
            batch_vectors: np.ndarray = self.get_embeddings(batch_texts)
            
            # 2. Prepare Weaviate Data Objects
            weaviate_objects: List[weaviate.classes.DataObject] = []
            for j in range(len(batch_data)):
                chunk = batch_data[j]
                vector = batch_vectors[j].tolist()
                
                # Create a Weaviate DataObject with properties and the pre-computed vector
                obj = weaviate.classes.DataObject(
                    properties={
                        "document_id": chunk["document_id"],
                        "chunk_text": chunk["chunk_text"],
                        "source_url": chunk["source_url"],
                        "chunk_index": chunk["chunk_index"],
                    },
                    vector=vector
                )
                weaviate_objects.append(obj)

            # 3. Import the batch to Weaviate
            collection = self.client.collections.get(self.collection_name)
            # Use the .insert_many() method for efficient batching
            response = collection.data.insert_many(weaviate_objects)

            if response.errors:
                print(f"\n--- Batch {i//batch_size} encountered errors ---")
                for error in response.errors:
                    print(f"  Error: {error.message}")
                print("-------------------------------------------\n")

        print("\nData ingestion complete.")
        count = self.client.collections.get(self.collection_name).aggregate.over_all(total_count=True).total_count
        print(f"Total objects in Weaviate collection '{self.collection_name}': {count}")


if __name__ == "__main__":
    
    # --- Execution ---
    ingestor = WeaviateIngestor(
        url=weaviate_url,
        model_id=model_id,
        collection_name=collection_name,
        show_progress=True
    )
    
    try:
        if not ingestor.client.is_live():
            print(f"ERROR: Could not connect to Weaviate at {weaviate_url}. Ensure Docker is running.")
        else:
            print(f"Successfully connected to Weaviate at {weaviate_url}")
            ingestor.create_schema()
            ingestor.ingest_data(dummy_chunks)
            
            # Example search (Hybrid Search combining keyword and semantic)
            print("\n--- Example Hybrid Search (Keyword + Semantic) ---")
            collection = ingestor.client.collections.get(collection_name)
            
            # The query text will be used for both keyword search and for generating a vector
            # (which you would do using the same get_embeddings() method as above)
            query_text = "What is the best way to handle large scale vector data?"
            query_vector = ingestor.get_embeddings([query_text])[0].tolist() # Generate query vector
            
            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector, # Pass the vector explicitly for the semantic part
                alpha=0.5, # 0.5 balances keyword and vector search equally
                limit=3,
                filters=weaviate.classes.Filter.by_property("source_url").equal("https://weaviate.io/docs/performance") # Example metadata filter
            )
            
            print(f"Query: '{query_text}' (Filtered by source_url)")
            for item in response.objects:
                # The returned object has the score, the text, and the metadata properties
                print(f"  Score: {item.metadata.score:.4f} | Source: {item.properties['source_url']} | Text: {item.properties['chunk_text']}")

    except Exception as e:
        print(f"An error occurred: {e}")