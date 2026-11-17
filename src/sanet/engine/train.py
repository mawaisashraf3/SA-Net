"""Real dataset training, validation-only selection, safe resume and experiment tracking."""

import logging
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sanet.config import Config
from sanet.data.audit import audit_manifest
from sanet.data.dataset import LaneDataset, collate, to_device
from sanet.engine.checkpoint import load_checkpoint, save_checkpoint
from sanet.engine.tracking import Ledger, environment, seed_everything, worker_seed
from sanet.io import write_json, write_jsonl
from sanet.models.losses import LaneLoss
from sanet.models.network import SANet

logger = logging.getLogger(__name__)


def make_loader(cfg, split, augment=False, generator=None):
    return DataLoader(
        LaneDataset(cfg, split, augment),
        batch_size=cfg.train.batch_size,
        shuffle=augment,
        num_workers=cfg.data.workers,
        collate_fn=collate,
        worker_init_fn=worker_seed,
        generator=generator,
        pin_memory=cfg.train.device.startswith("cuda"),
    )


def epoch_pass(model, criterion, loader, cfg, optimizer=None, scaler=None, warmup=False):
    training = optimizer is not None
    model.train(training)
    totals, count, started = {}, 0, time.perf_counter()
    for batch in loader:
        batch = to_device(batch, cfg.train.device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=cfg.train.amp):
                output = model(batch, warmup=warmup)
                losses = criterion(output, batch["targets"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Nonfinite loss for {batch['ids']}")
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip, error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
        n = len(batch["images"])
        count += n
        for k, value in losses.items():
            totals[k] = totals.get(k, 0.0) + float(value.detach()) * n
    return {
        **{k: v / count for k, v in totals.items()},
        "samples": count,
        "seconds": time.perf_counter() - started,
    }


def train(cfg: Config, output_dir, resume=None, stop_after=None):
    """stop_after supports interruption/recovery checks without changing the LR schedule."""
    cfg.validate()
    if stop_after is not None and not 1 <= stop_after <= cfg.train.epochs:
        raise ValueError("stop_after must be between 1 and the configured epoch count")
    if cfg.train.amp and not cfg.train.device.startswith("cuda"):
        raise ValueError("AMP is supported only on CUDA in this implementation")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and resume is None:
        raise FileExistsError(
            "Run already has a checkpoint; use --resume or a new output directory"
        )
    seed_everything(cfg.train.seed, cfg.train.threads)
    audit = audit_manifest(cfg.data.manifest, cfg.data.root, cfg.model.mode)
    generator = torch.Generator().manual_seed(cfg.train.seed)
    train_loader = make_loader(cfg, "train", True, generator)
    val_loader = make_loader(cfg, "val", False)
    model = SANet(cfg.model).to(cfg.train.device)
    criterion = LaneLoss(cfg).to(cfg.train.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)
    history, first_epoch, best = [], 0, math.inf
    if resume:
        state = load_checkpoint(resume, cfg.train.device)
        if (
            state["config"] != cfg.to_dict()
            or state["manifest_sha256"] != audit["manifest_sha256"]
            or state["dataset_sha256"] != audit["dataset_sha256"]
        ):
            raise ValueError("Resume requires identical configuration, manifest and asset bytes")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        random.setstate(state["python_rng"])
        generator.set_state(state["loader_rng"].cpu())
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        history, first_epoch, best = state["history"], state["epoch"] + 1, state["best"]
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "environment.json", environment())
    write_json(out / "data-audit.json", audit)
    ledger = Ledger(out.parent / "experiments.sqlite")
    run_id = str(out.resolve())
    ledger.start(run_id, cfg.to_dict(), cfg.data.manifest)
    try:
        until = min(cfg.train.epochs, stop_after) if stop_after is not None else cfg.train.epochs
        for epoch in range(first_epoch, until):
            lr = optimizer.param_groups[0]["lr"]
            train_metrics = epoch_pass(
                model,
                criterion,
                train_loader,
                cfg,
                optimizer,
                scaler,
                epoch < cfg.train.warmup_epochs,
            )
            val_metrics = epoch_pass(model, criterion, val_loader, cfg)
            scheduler.step()
            row = {"epoch": epoch, "learning_rate": lr, "train": train_metrics, "val": val_metrics}
            history.append(row)
            improved = val_metrics["total"] < best
            best = min(best, val_metrics["total"])
            write_jsonl(out / "history.jsonl", history)
            ledger.epoch(run_id, epoch, row)
            save_checkpoint(
                out / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                epoch,
                best,
                history,
                audit["manifest_sha256"],
                generator,
                audit["dataset_sha256"],
            )
            if improved:
                save_checkpoint(
                    out / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    cfg,
                    epoch,
                    best,
                    history,
                    audit["manifest_sha256"],
                    generator,
                    audit["dataset_sha256"],
                )
            logger.info(
                "epoch=%d train=%.5f val=%.5f lr=%.7f",
                epoch,
                train_metrics["total"],
                val_metrics["total"],
                lr,
            )
        status = "completed" if until == cfg.train.epochs else "interrupted_for_resume"
        write_json(
            out / "status.json",
            {
                "status": status,
                "completed_epochs": len(history),
                "selected_by": "minimum validation total loss",
                "best_validation_loss": best,
            },
        )
        ledger.finish(run_id, status)
    except BaseException:
        ledger.finish(run_id, "failed")
        raise
    finally:
        ledger.close()
    return history
