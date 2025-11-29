from dotenv import load_dotenv
import os
import logging
from config import *

# Load just the paths file (via ENV_PATHS or fallback)
# load_dotenv(dotenv_path=os.getenv("ENV_PATHS", "./configs/.env.paths"))

# log_dir = os.environ.get('index_log_dir')
if not log_dir:
    raise ValueError("Missing required environment variable 'index_log_dir' from .env.paths")
os.makedirs(log_dir, exist_ok=True)

def get_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, filename))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger