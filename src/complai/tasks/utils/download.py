import logging
import urllib.request
from pathlib import Path


logger = logging.getLogger(__name__)


def ensure_vader_lexicon(vader_lexicon_path: Path, cache_dir: Path) -> None:
    if not vader_lexicon_path.exists():
        try:
            logger.info("Downloading VADER lexicon...")
            from nltk.downloader import download as nltk_download

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


def ensure_nltk_wordnet(nltk_data_dir: Path) -> None:
    """Ensure NLTK WordNet data is downloaded."""
    wordnet_path = nltk_data_dir / "corpora" / "wordnet.zip"
    omw_path = nltk_data_dir / "corpora" / "omw-1.4.zip"
    if not wordnet_path.exists() or not omw_path.exists():
        try:
            logger.info("MMLU-Pro Robustness: Downloading NLTK data...")
            import nltk

            nltk.download("wordnet", download_dir=nltk_data_dir, quiet=False)
            nltk.download("omw-1.4", download_dir=nltk_data_dir, quiet=False)
        except Exception as e:
            raise RuntimeError(
                f"MMLU-Pro Robustness: Failed to download NLTK WordNet data: {e}"
            )


def ensure_wordnet_synonyms(synonyms_path: Path) -> None:
    """Ensure WordNet synonyms JSON file is downloaded."""
    if not synonyms_path.exists():
        try:
            logger.info("MMLU-Pro Robustness: Downloading WordNet synonyms...")
            synonyms_path.parent.mkdir(parents=True, exist_ok=True)

            url = "https://storage.googleapis.com/crfm-helm-public/source_datasets/augmentations/synonym_perturbation/wordnet_synonyms.json"

            urllib.request.urlretrieve(url, synonyms_path)
        except Exception as e:
            raise RuntimeError(
                f"MMLU-Pro Robustness: Failed to download WordNet synonyms: {e}. "
                f"You can manually download from {url} "
                f"and place it at {synonyms_path}"
            )


def ensure_dialect_mapping(dialect_mapping_path: Path) -> None:
    """Ensure dialect mapping file is downloaded."""
    if not dialect_mapping_path.exists():
        try:
            logger.info("MMLU-Pro Robustness: Downloading dialect mapping...")
            dialect_mapping_path.parent.mkdir(parents=True, exist_ok=True)

            url = "https://storage.googleapis.com/crfm-helm-public/source_datasets/augmentations/dialect_perturbation/SAE_to_AAVE_mapping.json"

            urllib.request.urlretrieve(url, dialect_mapping_path)
        except Exception as e:
            raise RuntimeError(
                f"MMLU-Pro Robustness: Failed to download dialect mapping: {e}. "
                f"Please manually download the file from {url} "
                f"and place it at {dialect_mapping_path}"
            )


def ensure_punkt_tab_tokenizer(punkt_tab_tokenizer_path: Path, cache_dir: Path) -> None:
    if not punkt_tab_tokenizer_path.exists():
        try:
            logger.info("Downloading nltk data...")
            from nltk.data import path as nltk_data_path
            from nltk.downloader import download

            download("punkt_tab", download_dir=cache_dir)
            nltk_data_path.append(str(cache_dir))

        except Exception as e:
            raise RuntimeError(f"Failed to download NLTK tokenizer data: {e}. ")
