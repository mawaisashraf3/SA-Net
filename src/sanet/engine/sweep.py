"""Validation-only hyperparameter search and module ablations."""

from copy import deepcopy
from itertools import product
from pathlib import Path

from sanet.engine.train import train
from sanet.io import write_json


def sweep(cfg, output_dir, learning_rates, weight_decays, seeds):
    root, results = Path(output_dir), []
    for i, (lr, decay, seed) in enumerate(product(learning_rates, weight_decays, seeds)):
        trial = deepcopy(cfg)
        trial.train.learning_rate, trial.train.weight_decay, trial.train.seed = (
            float(lr),
            float(decay),
            int(seed),
        )
        history = train(trial, root / f"trial-{i:03d}")
        results.append(
            {
                "trial": i,
                "learning_rate": lr,
                "weight_decay": decay,
                "seed": seed,
                "best_validation_loss": min(h["val"]["total"] for h in history),
            }
        )
    if not results:
        raise ValueError("Empty search space")
    selected = min(results, key=lambda r: r["best_validation_loss"])
    write_json(
        root / "search.json",
        {
            "trials": results,
            "selected": selected,
            "selection": "validation loss only; test never evaluated",
        },
    )
    return selected


def ablate(cfg, output_dir, seeds):
    variants = (
        {
            "backbone_head": (False, False, False),
            "plus_ampm": (False, True, False),
            "plus_ampm_era": (False, True, True),
        }
        if cfg.model.mode == "image"
        else {
            "dense_lift": (False, False, False),
            "sfm": (True, False, False),
            "sfm_ampm": (True, True, False),
            "sfm_ampm_era": (True, True, True),
        }
    )
    root, results = Path(output_dir), []
    for name, flags in variants.items():
        for seed in seeds:
            trial = deepcopy(cfg)
            trial.model.sfm, trial.model.ampm, trial.model.era = flags
            trial.train.seed = int(seed)
            run = root / f"{name}-seed-{seed}"
            history = train(trial, run)
            results.append(
                {
                    "variant": name,
                    "seed": seed,
                    "run": str(run),
                    "best_validation_loss": min(h["val"]["total"] for h in history),
                }
            )
    write_json(
        root / "ablation.json",
        {
            "results": results,
            "reproduction_claim": False,
            "note": "Independent backbone baseline, not DLNet; image mode has no multimodal SFM",
        },
    )
    return results
