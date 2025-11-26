
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Set
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import tempfile
import json
import shutil
import time
import os

# from locate_or_get_chunk_from_pdf import locate_sections_parallel
from markers_check import uses_numbered_citations
from sections_division import chunk_pdf_with_grobid
from map_refs_by_markers import match_refs_by_marker
from ref_marker_tailing import parse_mark_refs
from find_refs_by_grobid import find_refs
from config import *

Sections = List[Dict[str, Any]]
Ref = List[Dict[str, Any]]
PreparedPaper = Tuple[Sections, Ref, bool, str, str]

def get_pdf_paths(folder_path):
    folder = Path(folder_path)
    # Use glob to find all PDF files recursively with .pdf suffix (case-sensitive)
    pdf_files = [str(path) for path in folder.glob("**/*.pdf")]
    return pdf_files

def extract_arxiv_id(input_str: str) -> str:
    # Check if input is a URL
    if input_str.startswith("http://") or input_str.startswith("https://"):
        # Parse the URL to get the path part
        parsed_url = urlparse(input_str)
        # The arXiv ID is the last segment of the path, e.g. '1409.0473v7'
        arxiv_id = parsed_url.path.split('/')[-1]
        return arxiv_id
    else:
        # Assume input is a filepath, extract filename without extension
        filename = os.path.basename(input_str)
        arxiv_id = os.path.splitext(filename)[0]
        return arxiv_id

# GROBID with multithreading
def prepare_multithreaded(pdf_path: str) -> PreparedPaper:
    sections= chunk_pdf_with_grobid(pdf_path)
    use_markers = uses_numbered_citations(pdf_path)
    seen_refs = set()
    arxiv_id = extract_arxiv_id(pdf_path)
    if use_markers:
        for secs in sections:
            refs = match_refs_by_marker(pdf_path, secs, as_list=True)['reference_mapping'] # secs =)))))
            seen_refs.update(refs)
        refs_tailed = parse_mark_refs(seen_refs)
        return sections, refs_tailed, use_markers, pdf_path, arxiv_id
    else:
        refs_content = find_refs(pdf_path)
        return (sections, refs_content, use_markers, pdf_path, arxiv_id)

def _atomic_write_json(path: str, data: Any) -> None:
    """
    Write JSON atomically: write to a temp file in the same directory then replace.
    Uses json.dump(..., default=str) to avoid failures for non-serializable objects.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_grobid_", dir=dir_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, path)
    finally:
        # If something went wrong and tmp file still exists, attempt cleanup
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def _load_checkpoint(path: str) -> List[Tuple[Any, Any, Any, str, str]]:
    """
    Load checkpoint JSON if available. If corrupted, move the broken file aside and return [].
    Expected format: list of 5-tuples/lists: (sections, refs, use_markers, pdf_path, arxiv_id)
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Ensure we have a list of 5-element items; tolerate lists of lists (from json)
        cleaned: List[Tuple[Any, Any, Any, str, str]] = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 5:
                # cast to tuple of length 5: (sections, refs, use_markers, pdf_path, arxiv_id)
                cleaned.append((item[0], item[1], item[2], item[3], item[4]))
        return cleaned
    except Exception as e:
        # Backup corrupted file and start fresh
        try:
            backup_path = f"{path}.corrupt.{int(time.time())}"
            shutil.move(path, backup_path)
            print(f"Warning: checkpoint file was corrupted and has been moved to {backup_path}. Starting from scratch.")
        except Exception as move_err:
            print(f"Warning: failed to back up corrupted checkpoint ({move_err}). Starting from scratch.")
        return []

def _extract_processed_sets(prepared_secs: List[Tuple[Any, Any, Any, str, str]]) -> Tuple[Set[str], Set[str]]:
    """
    From prepared_secs build:
      - set of processed PDF absolute paths
      - set of processed arXiv IDs (if available and non-empty)
    """
    processed_paths: Set[str] = set()
    processed_arxiv: Set[str] = set()
    for entry in prepared_secs:
        # entry format: (sections, refs, use_markers, pdf_path, arxiv_id)
        _, _, _, pdf_path, arxiv_id = entry
        if isinstance(pdf_path, str) and pdf_path:
            processed_paths.add(os.path.abspath(pdf_path))
            # also add basename for loose matching (helps if stored relative paths differ)
            processed_paths.add(os.path.abspath(os.path.basename(pdf_path)))
        if isinstance(arxiv_id, str) and arxiv_id:
            processed_arxiv.add(arxiv_id)
    return processed_paths, processed_arxiv

def grobid_processing(show_progress: bool = True) -> None:
    """
    Resume GROBID processing using intermediate checkpoint `grobid_prepared_path`.
    Skips PDFs already present in the checkpoint (by pdf_path or arXiv id).
    Saves intermediate results every `grobid_save_every` completed tasks.
    """
    start_time: float = time.time()

    # Load existing prepared entries (if any) from checkpoint
    prepared_secs: List[Tuple[Any, Any, Any, str, str]] = _load_checkpoint(grobid_prepared_path)
    processed_paths, processed_arxiv = _extract_processed_sets(prepared_secs)

    # All pdfs discovered in the folder
    pdf_paths: List[str] = get_pdf_paths(pdfs_folder)
    if not pdf_paths:
        print("No PDFs found in folder.")
        return

    # Build list of pdfs to process: skip those already processed (match by absolute path, basename, or arXiv id if possible)
    remaining_pdf_paths: List[str] = []
    for p in pdf_paths:
        abs_p = os.path.abspath(p)
        basename_p = os.path.abspath(os.path.basename(p))
        if abs_p in processed_paths or basename_p in processed_paths:
            continue
        remaining_pdf_paths.append(p)

    if not remaining_pdf_paths:
        print("All discovered PDFs are already processed according to the checkpoint. Nothing to do.")
        print(f"Total time: {time.time() - start_time:.2f}s (no new files)")
        return

    print(f"Resuming: {len(prepared_secs)} already saved entries, {len(remaining_pdf_paths)} to process now.")

    # Start processed_count from the already-saved count so periodic saves remain on schedule
    processed_count: int = len(prepared_secs)

    with ThreadPoolExecutor(max_workers=grobid_max_workers) as executor:
        futures = [executor.submit(prepare_multithreaded, pdf_path) for pdf_path in remaining_pdf_paths]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Processing PDFs (resuming)")

        for future in iterator:
            try:
                sections, refs, use_markers, pdf_path, arxiv_id = future.result()
            except Exception as e:
                # Count failures as completed so periodic saving remains on schedule.
                print(f"Task generated an exception: {e}")
            else:
                # Append new successful result
                prepared_secs.append((sections, refs, use_markers, pdf_path, arxiv_id))
            finally:
                processed_count += 1
                # Periodic save whenever processed_count reaches another multiple of grobid_save_every
                if grobid_save_every > 0 and (processed_count % grobid_save_every) == 0:
                    try:
                        _atomic_write_json(grobid_prepared_path, prepared_secs)
                        print(f"Intermediate save after {processed_count} completed tasks.")
                    except Exception as e:
                        print(f"Intermediate save failed after {processed_count} tasks: {e}")

    # Final atomic save
    try:
        _atomic_write_json(grobid_prepared_path, prepared_secs)
        print("Final prepared papers file saved.")
    except Exception as e:
        print(f"Failed to write final prepared file: {e}")

    end_time = time.time()
    print(f"Total time taken (resume run): {end_time - start_time:.2f} seconds")


# def grobid_processing(show_progress: bool = True) -> None:
#     """
#     Process PDFs concurrently and save intermediate results every `grobid_save_every`
#     completed tasks (including tasks that failed). Uses tqdm when show_progress is True.

#     Parameter
#     ----------
#     show_progress : bool
#         If True, show a tqdm progress bar.
#     """
    
#     start_time: float = time.time()
#     prepared_secs: List[Tuple[Any, Any, Any, str, str]] = []
#     pdf_paths: List[str] = get_pdf_paths(pdfs_folder)

#     if not pdf_paths:
#         print("No PDFs found.")
#         return

#     with ThreadPoolExecutor(max_workers=grobid_max_workers) as executor:
#         futures = [executor.submit(prepare_multithreaded, pdf_path) for pdf_path in pdf_paths]
#         processed_count: int = 0

#         iterator = as_completed(futures)
#         if show_progress:
#             iterator = tqdm(iterator, total=len(futures), desc="Processing PDFs")

#         for future in iterator:
#             try:
#                 sections, refs, use_markers, pdf_path, arxiv_id = future.result()
#             except Exception as e:
#                 # Log and continue; we still count this completed task so periodic saves are on-schedule.
#                 print(f"Task generated an exception: {e}")
#             else:
#                 prepared_secs.append((sections, refs, use_markers, pdf_path, arxiv_id))
#             finally:
#                 processed_count += 1
#                 if grobid_save_every > 0 and (processed_count % grobid_save_every) == 0:
#                     # atomic write so we don't leave a partially-written JSON file
#                     try:
#                         _atomic_write_json(grobid_prepared_path, prepared_secs)
#                     except Exception as e:
#                         print(f"Failed to write intermediate save after {processed_count} tasks: {e}")

#     end_time = time.time()
#     print(f"Total time taken for GROBIDing {len(pdf_paths)} pdfs: {end_time - start_time:.2f} seconds")

#     # final save (atomic)
#     try:
#         _atomic_write_json(grobid_prepared_path, prepared_secs)
#         print("Prepared papers are saved!")
#     except Exception as e:
#         print(f"Failed to write final prepared file: {e}")