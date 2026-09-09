import os
import tempfile
from pathlib import Path

def atomic_write(path: Path, content: str) -> None:
    """Write text through a temporary file and atomic replacement."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
import json
import hashlib
from typing import Any

def digest_json(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON value."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
import math
import numpy as np
def json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values to standard JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
