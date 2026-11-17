"""Atomic training checkpoints with RNG/sampler/optimizer state and input hash."""

import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

from sanet.config import from_dict
from sanet.models.network import SANet


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    cfg,
    epoch,
    best,
    history,
    manifest_hash,
    generator,
    dataset_hash,
):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    os.close(fd)
    state = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg.to_dict(),
        "epoch": epoch,
        "best": best,
        "history": history,
        "manifest_sha256": manifest_hash,
        "dataset_sha256": dataset_hash,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "loader_rng": generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return state


def restore_model(path, device="cpu"):
    state = load_checkpoint(path, device)
    cfg = from_dict(state["config"])
    spec = deepcopy(cfg.model)
    spec.pretrained = False  # Checkpoint already supplies all weights; no implicit downloads.
    model = SANet(spec).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model, cfg, state
