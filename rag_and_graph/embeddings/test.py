# import json
# import tiktoken
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union, Optional, Iterable, Generator
import random
import string

# encoding_name = "cl100k_base"
# JSONL_PATH = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/data/processed_papers_json/rechunked_final_papers.jsonl"
MODEL_PATH = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/rag_and_graph/embeddings/qwen3_0_6B_model"

# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens


# with open(JSONL_PATH, 'r', encoding='utf-8') as file:
#     for line in file:
#         try:
#             data = json.loads(line.strip())  # Parse each line as dict/list
#             token_len = num_tokens_from_string(data['abstract'], encoding_name)
#             if token_len > 400:
#                 print(token_len)
#         except json.JSONDecodeError:
#             print("Skipping invalid line")


class GpuTest:
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
        return SentenceTransformer(MODEL_PATH, device=self.device)

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

    def generate_dummy_texts(self, num_texts: int = 1000, min_length: int = 50, max_length: int = 200) -> list[str]:
        def random_string(length: int) -> str:
            return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
        
        return [random_string(random.randint(min_length, max_length)) for _ in range(num_texts)]


if __name__ == "__main__":
    # Generate 500 texts (~100k total tokens) - adjust num_texts for your GPU capacity
    tester = GpuTest(batch_size=16)
    dummy_texts = tester.generate_dummy_texts(num_texts=500, min_length=250, max_length=1024)
    
    # Test your batch_encode function
    embeddings = tester.batch_encode(dummy_texts)
    print(f"Generated {len(embeddings)} embeddings")