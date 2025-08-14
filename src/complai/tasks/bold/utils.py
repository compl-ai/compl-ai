import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def ensure_vader_lexicon(vader_lexicon_path: Path, cache_dir: Path) -> None:
    if not vader_lexicon_path.exists():
        logger.info("Downloading VADER lexicon...")
        from nltk.downloader import download as nltk_download

        try:
            nltk_download("vader_lexicon", download_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to download VADER lexicon: {e}. ")


def ensure_word2vec_weights(word_2_vec_path: Path, cache_dir: Path) -> None:
    if not word_2_vec_path.exists():
        try:
            logger.info("Downloading Word2Vec weights...")

            import gdown

            gdown.download(
                id="19um3Uu9m0AcsynwuKvpntO80LpX6oFqM",
                output=f"{cache_dir}/",
                quiet=False,
                resume=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Word2Vec weights: {e}. "
                "You can manually download the file from "
                "https://drive.google.com/file/d/19um3Uu9m0AcsynwuKvpntO80LpX6oFqM/view?usp=sharing"
                "and place it at "
                f"{word_2_vec_path}"
            )
