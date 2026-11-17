import json
import sqlite3
from pathlib import Path

import pytest
import torch

from sanet.engine.checkpoint import load_checkpoint, restore_model
from sanet.engine.evaluate import evaluate
from sanet.engine.train import train
from sanet.evaluation.report import report_run


def test_checkpoint_resume_matches_uninterrupted(config, records, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(config, full)
    train(config, resumed, stop_after=1)
    train(config, resumed, resume=resumed / "last.pt")
    a, b = load_checkpoint(full / "last.pt"), load_checkpoint(resumed / "last.pt")
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    assert a["scheduler"] == b["scheduler"]
    assert len(b["history"]) == 2
    assert json.loads((resumed / "status.json").read_text())["status"] == "completed"
    db = sqlite3.connect(tmp_path / "experiments.sqlite")
    assert db.execute("SELECT count(*) FROM epochs").fetchone()[0] == 4
    db.close()
    model, cfg, _ = restore_model(full / "best.pt")
    assert cfg.model.backbone == "compact"
    assert model.training  # caller controls eval state
    metrics = evaluate(
        full / "best.pt", config.data.manifest, config.data.root, "test", tmp_path / "eval"
    )
    assert metrics["samples"] == 1 and not metrics["reproduction_claim"]
    report_run(full, tmp_path / "eval", tmp_path / "report")
    assert (tmp_path / "report/learning-curves.png").is_file()
    assert (tmp_path / "report/epochs.csv").is_file()
    with pytest.raises(FileExistsError):
        train(config, full)
    config.train.learning_rate *= 2
    with pytest.raises(ValueError, match="identical"):
        train(config, resumed, resume=resumed / "last.pt")


def test_bev_train_evaluate(config, records, tmp_path):
    config.model.mode = "bev"
    config.train.epochs = 1
    train(config, tmp_path / "bev")
    metrics = evaluate(
        tmp_path / "bev/best.pt",
        config.data.manifest,
        config.data.root,
        "test",
        tmp_path / "bev-eval",
    )
    assert "diagnostic_3d_at_0.5m" in metrics
    assert Path(tmp_path / "bev-eval/predictions.jsonl").exists()
