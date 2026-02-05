from pathlib import Path

from platformdirs import user_cache_dir


CACHE_DIR = Path(user_cache_dir("complai"))
