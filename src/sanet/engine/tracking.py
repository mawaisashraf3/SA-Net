"""Local experiment ledger, deterministic state and auditable environment capture."""

import hashlib
import json
import platform
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from sanet.io import sha256


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def worker_seed(worker_id):
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def environment():
    import cv2
    import scipy
    import torchvision

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = None
    source_root = Path(__file__).resolve().parents[1]
    source_digest = hashlib.sha256()
    for source_file in sorted(source_root.rglob("*.py")):
        source_digest.update(str(source_file.relative_to(source_root)).encode())
        source_digest.update(source_file.read_bytes())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "git_commit": commit,
        "python_source_sha256": source_digest.hexdigest(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "torch_threads": torch.get_num_threads(),
    }


class Ledger:
    """SQLite records runs and epoch metrics; data assets remain on disk."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, config_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL, started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS epochs (
                run_id TEXT NOT NULL REFERENCES runs(id), epoch INTEGER NOT NULL,
                metrics_json TEXT NOT NULL, PRIMARY KEY(run_id, epoch)
            );
        """)
        self.db.commit()

    def start(self, run_id, config, manifest):
        self.db.execute(
            "INSERT INTO runs(id,status,config_json,manifest_sha256) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status='running',updated_at=CURRENT_TIMESTAMP",
            (run_id, "running", json.dumps(config), sha256(manifest)),
        )
        self.db.commit()

    def epoch(self, run_id, epoch, metrics):
        self.db.execute(
            "INSERT OR REPLACE INTO epochs VALUES(?,?,?)",
            (run_id, epoch, json.dumps(metrics, allow_nan=False)),
        )
        self.db.commit()

    def finish(self, run_id, status):
        self.db.execute(
            "UPDATE runs SET status=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (status, run_id)
        )
        self.db.commit()

    def close(self):
        self.db.close()
