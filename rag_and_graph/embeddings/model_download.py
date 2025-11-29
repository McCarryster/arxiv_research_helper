from huggingface_hub import snapshot_download
from tqdm import tqdm
from config import *

def download_qwen_embedding_model(
    local_dir: str,
    model_id: str,
    show_progress: bool = True
) -> str:
    """
    Downloads the Qwen embedding model from Hugging Face to a specified local directory.

    Args:
        local_dir: The path to the directory where the model files will be saved.
        model_id: The Hugging Face model ID (default is Qwen/Qwen3-Embedding-0.6B).
        show_progress: Flag to show or hide the download progress bar.

    Returns:
        The path to the local directory where the model was downloaded.
    """
    print(f"Attempting to download model '{model_id}' to: {local_dir}")

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # Important for a clean local copy
            tqdm_class=tqdm if show_progress else None # type: ignore
        )
        print(f"Model downloaded successfully to: {local_path}")
        return local_path
    except Exception as e:
        print(f"An error occurred during download: {e}")
        raise

download_qwen_embedding_model(model_path, model_id)