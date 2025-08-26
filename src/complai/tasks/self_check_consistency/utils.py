import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def ensure_nltk_data(nltk_tokenizer_path: Path, cache_dir: Path) -> None:
    if not nltk_tokenizer_path.exists():
        logger.info("Downloading nltk data...")
        try:
            from nltk.downloader import download as nltk_download

            nltk_download("punkt_tab", download_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to download NLTK tokenizer data: {e}. ")
