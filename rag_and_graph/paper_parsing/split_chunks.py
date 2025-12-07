from typing import List, Dict, Any, Optional
import hashlib
import uuid
import json
from tqdm import tqdm
import tiktoken
import math

# -------------------------
# Helper functions
# -------------------------
def generate_checksum(text: str) -> str:
    """
    Create a sha256 checksum for the given text.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text.encode("utf-8"))
    return sha256_hash.hexdigest()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Returns the number of tokens in a text string.
    Uses tiktoken when available, otherwise falls back to whitespace token count.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


# -------------------------
# Main functions
# -------------------------
def process_single_chunk(
    original: Dict[str, Any],
    paper_id: str,
    max_tokens_per_chunk: int = 1024,
    encoding_name: str = "cl100k_base",
) -> List[Dict[str, Any]]:
    """
    Split a single chunk dict's 'chunk_text' into multiple chunk dicts.
    This distributes tokens evenly so there won't be a tiny 1-token tail.

    Returns a list of dicts similar to `original` (preserving metadata fields),
    but with new chunk_ids and checksums. The returned chunk_num values are
    placeholders (the caller reindexes them).
    """
    text: str = original.get("chunk_text", "")
    if not isinstance(text, str):
        raise TypeError("original['chunk_text'] must be a string")

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        raise ValueError(f"Could not get tiktoken encoding '{encoding_name}': {exc}")

    tokens: List[int] = encoding.encode(text)
    total_tokens: int = len(tokens)

    # If already small enough, return a single dict
    if total_tokens <= max_tokens_per_chunk:
        chunk_text = encoding.decode(tokens)
        token_len = num_tokens_from_string(chunk_text, encoding_name)
        new_chunk: Dict[str, Any] = {
            "chunk_num": int(original.get("chunk_num", 1)),
            "chunk_paper_id": paper_id,
            "chunk_id": str(uuid.uuid4()),
            "section_title": original.get("section_title", ""),
            "chunk_text": chunk_text,
            "token_len": token_len,
            "is_citation_context": original.get("is_citation_context", False),
            "citations": list(original.get("citations", [])),
            "checksum": generate_checksum(chunk_text),
            "uses_markers": original.get("uses_markers", False),
            "chunk_embedding_id": "",
        }
        return [new_chunk]

    # Determine number of chunks and balanced chunk size:
    n_chunks: int = math.ceil(total_tokens / max_tokens_per_chunk)
    chunk_size: int = math.ceil(total_tokens / n_chunks)
    # Safety check (should be true by construction)
    if chunk_size > max_tokens_per_chunk:
        chunk_size = max_tokens_per_chunk

    chunks: List[Dict[str, Any]] = []
    for i in range(0, total_tokens, chunk_size):
        token_slice = tokens[i : i + chunk_size]
        chunk_text = encoding.decode(token_slice)
        token_len = num_tokens_from_string(chunk_text, encoding_name)

        chunk_dict: Dict[str, Any] = {
            "chunk_num": int(original.get("chunk_num", 1)),  # caller will reindex
            "chunk_paper_id": paper_id,
            "chunk_id": str(uuid.uuid4()),
            "section_title": original.get("section_title", ""),
            "chunk_text": chunk_text,
            "token_len": token_len,
            "is_citation_context": original.get("is_citation_context", False),
            "citations": list(original.get("citations", [])),
            "checksum": generate_checksum(chunk_text),
            "uses_markers": original.get("uses_markers", False),
            "chunk_embedding_id": "",
        }
        chunks.append(chunk_dict)

    return chunks

def process_jsonl_stream(
    input_path: str,
    output_path: str,
    max_tokens_per_chunk: int = 1024,
    encoding_name: str = "cl100k_base",
    show_progress: bool = True,
) -> None:
    """
    Stream through an input JSONL file line-by-line, re-chunk any 'chunks' list
    found in each JSON object, and write the processed object (one per line)
    to the output JSONL file.

    Each input line is parsed as JSON. If it contains a 'chunks' list, each original
    chunk is passed to process_single_chunk and the combined resulting chunks
    replace the original 'chunks' list. The resulting object's 'chunks' list is
    reindexed sequentially (1-based) per document before writing.

    If a line is invalid JSON, it's skipped with an error message (but processing continues).
    """
    # Open input and output; stream line-by-line
    with open(input_path, "r", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        iterator = in_f if not show_progress else tqdm(in_f, desc="Processing JSONL", unit="lines")
        for line_num, line in enumerate(iterator, start=1):
            line = line.strip()
            if not line:
                # skip empty lines
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping line {line_num}: JSON decode error: {exc}")
                continue

            # Find 'chunks' list in the object and process if present
            if isinstance(obj, dict) and "chunks" in obj and isinstance(obj["chunks"], list):
                original_chunks: List[Dict[str, Any]] = obj["chunks"]
                new_chunks: List[Dict[str, Any]] = []

                # iterate original chunks (use tqdm only if show_progress True and many chunks)
                inner_iterator = original_chunks if not show_progress else tqdm(original_chunks, desc=f"Rechunking doc line {line_num}", leave=False)
                for original in inner_iterator:
                    try:
                        splitted = process_single_chunk(
                            original,
                            obj['arxiv_id'],
                            max_tokens_per_chunk=max_tokens_per_chunk,
                            encoding_name=encoding_name,
                        )
                        new_chunks.extend(splitted)
                    except Exception as exc:
                        # Keep processing other chunks even if one chunk fails
                        print(f"Error processing chunk in line {line_num}: {exc}")
                        continue

                # Reindex chunk_num sequentially starting from 1 for this document
                for idx, ch in enumerate(new_chunks, start=1):
                    ch["chunk_num"] = idx

                obj["chunks"] = new_chunks

            # Write the (possibly modified) object as a single JSON line
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/data/processed_papers_json/final_papers.jsonl"
    output_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/data/processed_papers_json/rechunked_final_papers.jsonl"
    process_jsonl_stream(input_path, output_path)