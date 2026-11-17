"""Strict JSONL, content provenance and atomic artifact writes."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    atomic_text(path, "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            f.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path: str | Path, obj) -> None:
    atomic_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rooted(root: str | Path, path: str) -> Path:
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {path}")
    return result
