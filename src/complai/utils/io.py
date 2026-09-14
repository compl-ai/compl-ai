import os
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from complai.irt import FitResult

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

def check_output_available(
    output_dir: Path
) -> tuple[Path, Path]:
    """Return output paths if writing them is allowed."""
    output_dir = output_dir.expanduser().resolve()
    params_path = output_dir / "params.json"
    subset_path = output_dir / "subset.jsonl"

    existing = [path for path in (params_path, subset_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Output already exists: {existing[0]}")

    return params_path, subset_path

def write_outputs(
    result: "FitResult", output_dir: Path
) -> tuple[Path, Path]:
    """Write the fitted params and selected subset atomically."""
    params_path, subset_path = check_output_available(output_dir)
    output_dir = params_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Hybrid formatting: Pretty-print the top-level configuration metadata, but manually
    # serialize the massive columnar data array to keep exactly 1 record per line.
    items_obj = result.params.pop("items")
    base_json = json.dumps(result.params, indent=2, sort_keys=True, allow_nan=False)
    
    columns_json = json.dumps(items_obj["columns"])
    data_rows = [json.dumps(row, separators=(",", ":")) for row in items_obj["data"]]
    data_json = "[\n      " + ",\n      ".join(data_rows) + "\n    ]"
    items_json = f'{{\n    "columns": {columns_json},\n    "data": {data_json}\n  }}'
    
    params_text = base_json[:-2] + ',\n  "items": ' + items_json + '\n}\n'
    result.params["items"] = items_obj  # Restore in case object is reused
    subset_text = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in result.subset
    )
    atomic_write(params_path, params_text)
    try:
        atomic_write(subset_path, subset_text)
    except Exception:
        params_path.unlink(missing_ok=True)
        raise

    return params_path, subset_path

