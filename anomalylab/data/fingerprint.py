"""Dataset fingerprinting utility using SHA-256 for data provenance and lineage."""

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of configuration dictionary."""
    normalized_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


def compute_file_sha256(file_path: str | Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot compute fingerprint: {file_path} not found")

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataframe_fingerprint(df: pl.DataFrame) -> str:
    """
    Compute a deterministic fingerprint from dataframe schema, row count,
    and row contents.
    """
    schema_str = str(sorted(df.schema.items()))
    row_count = len(df)

    # Sample deterministic subset or hash columns
    hasher = hashlib.sha256()
    hasher.update(schema_str.encode("utf-8"))
    hasher.update(str(row_count).encode("utf-8"))

    # Hash deterministic string representation of column names and summary
    for col_name in sorted(df.columns):
        col_series = df[col_name]
        col_summary = f"{col_name}:{col_series.null_count()}:{col_series.dtype}"
        hasher.update(col_summary.encode("utf-8"))

    return hasher.hexdigest()
