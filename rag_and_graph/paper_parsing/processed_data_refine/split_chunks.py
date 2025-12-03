from __future__ import annotations
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import tiktoken


def generate_checksum(text: str) -> str:
    """
    Create a sha256 checksum for provided text.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text.encode("utf-8"))
    return sha256_hash.hexdigest()


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Returns the number of tokens in a text string for the given tiktoken encoding name.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _split_text_by_tokens(text: str, encoding: tiktoken.Encoding, max_tokens: int) -> List[str]:
    """
    Split text into a list of strings such that each string's token length (for the encoding)
    is <= max_tokens. This is done by encoding the entire text into token ids and slicing
    the token id list into segments of at most max_tokens, then decoding each slice.
    This preserves token boundaries (no mid-token chops).
    """
    token_ids: List[int] = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    parts: List[str] = []
    start = 0
    total = len(token_ids)
    while start < total:
        end = min(start + max_tokens, total)
        segment_ids = token_ids[start:end]
        segment_text = encoding.decode(segment_ids)
        parts.append(segment_text)
        start = end
    return parts


def process_jsonl(
    input_path: str,
    output_path: str,
    encoding_name: str = "cl100k_base",
    max_tokens: int = 1024,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    Process a JSONL file at input_path, split chunks with token_len > max_tokens into
    smaller token-aligned chunks, recompute chunk_num, chunk_id, token_len, checksum,
    and write the updated papers to output_path as JSONL.

    Parameters
    ----------
    input_path : str
        Path to input JSONL file (one JSON object per line).
    output_path : str
        Path where output JSONL will be written.
    encoding_name : str
        tiktoken encoding name to use for tokenization (default 'cl100k_base').
    max_tokens : int
        Maximum tokens allowed per chunk_text (default 1024).
    show_progress : bool
        If True, show tqdm progress bar while iterating input file lines.

    Returns
    -------
    dict
        statistics: {
            "papers_processed": int,
            "chunks_before": int,
            "chunks_after": int
        }
    """
    encoding = tiktoken.get_encoding(encoding_name)

    papers_processed = 0
    chunks_before_total = 0
    chunks_after_total = 0

    # Open files and iterate input line-by-line
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        iterator = infile
        if show_progress:
            iterator = tqdm(infile, desc="Papers", unit="paper")

        for raw_line in iterator:
            # raw_line = raw_line.strip()
            # if not raw_line:
            #     continue
            # try:
            #     paper: Dict[str, Any] = json.loads(raw_line)
            # except json.JSONDecodeError:
            #     # Skip invalid line but continue processing others
            #     continue

            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                paper_obj = json.loads(raw_line)
            except json.JSONDecodeError:
                # skip invalid JSON line
                continue

            # 🔥 NEW CHECK — skip if it's not a dict
            if not isinstance(paper_obj, dict):
                # example: line is a JSON string or array → skip
                continue

            papers_processed += 1

            original_chunks: List[Dict[str, Any]] = paper_obj.get("chunks", [])
            chunks_before_total += len(original_chunks)

            new_chunks: List[Dict[str, Any]] = []
            new_chunk_counter = 1  # chunk_num starts at 1 for each paper

            for orig_chunk in original_chunks:
                # Extract original text
                orig_text: str = orig_chunk.get("chunk_text", "") or ""
                # Token-split into pieces (token-aligned)
                pieces: List[str] = _split_text_by_tokens(orig_text, encoding, max_tokens)

                # For each piece create a new chunk dict copying over preserved fields
                for piece_text in pieces:
                    new_chunk: Dict[str, Any] = orig_chunk.copy()  # shallow copy
                    # Update only the allowed fields:
                    new_chunk["chunk_text"] = piece_text
                    new_chunk["token_len"] = num_tokens_from_string(piece_text, encoding_name)
                    # chunk_num sequential across resulting chunks for this paper
                    new_chunk["chunk_num"] = new_chunk_counter
                    new_chunk_counter += 1
                    # new unique chunk id (we keep it different to reflect new content boundaries)
                    new_chunk["chunk_id"] = uuid.uuid4().hex
                    # recompute checksum
                    new_chunk["checksum"] = generate_checksum(piece_text)

                    # Preserve other fields (citations, is_citation_context, section_title, etc.)
                    new_chunks.append(new_chunk)

            chunks_after_total += len(new_chunks)

            # Replace the paper's chunks with the new_chunks, leaving all other top-level fields alone
            paper_obj["chunks"] = new_chunks

            # (Optionally) recompute paper-level fields? The user asked not to touch anything else,
            # so we leave paper_embedding_id, citation_count, references, etc. untouched.

            # Dump updated paper as single-line JSON
            outfile.write(json.dumps(paper_obj, ensure_ascii=False) + "\n")

    stats: Dict[str, int] = {
        "papers_processed": papers_processed,
        "chunks_before": chunks_before_total,
        "chunks_after": chunks_after_total,
    }
    return stats



if __name__ == "__main__":
    input_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/data/processed_papers_json/final_paper_example.json"
    output_path = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/data/processed_papers_json/rechunked_paper_example.json"
    stats = process_jsonl(input_path, output_path)
    print(stats)