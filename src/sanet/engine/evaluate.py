"""Checkpoint evaluation/inference with measured latency, explicit protocols and exports."""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sanet.data.audit import audit_manifest
from sanet.data.dataset import LaneDataset, collate, to_device
from sanet.engine.checkpoint import restore_model
from sanet.engine.tracking import seed_everything
from sanet.evaluation.decode import decode, original_image_lanes
from sanet.evaluation.metrics import (
    Counts,
    grouped_bootstrap_f1,
    lane_3d_counts,
    lane_iou_counts,
    pixel_counts,
    pixel_metrics,
)
from sanet.evaluation.tusimple import evaluate_frame, sample_at_rows
from sanet.io import sha256, write_json, write_jsonl


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(checkpoint, manifest, root, split, output_dir, device="cpu", score=True):
    model, cfg, state = restore_model(checkpoint, device)
    cfg.data.manifest, cfg.data.root, cfg.train.device = str(manifest), str(root), device
    seed_everything(cfg.train.seed, cfg.train.threads)
    audit = audit_manifest(manifest, root, cfg.model.mode)
    # Checkpoint training manifest includes the registered held-out partition. Test
    # labels have never been used for optimization or checkpoint selection.
    dataset = LaneDataset(cfg, split)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=cfg.data.workers)
    model.eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, per_frame50, per_frame75, distances, tusimple = [], [], [], [], []
    total50, total75, pixels = Counts(), Counts(), np.zeros(4, dtype=np.int64)
    model_ms, pipeline_ms, groups = [], [], []
    for sample in loader:
        _sync(device)
        start = time.perf_counter()
        batch = to_device(sample, device)
        _sync(device)
        model_start = time.perf_counter()
        result = model(batch)
        _sync(device)
        model_elapsed = (time.perf_counter() - model_start) * 1000
        lanes = decode(result, cfg)[0]
        if cfg.model.mode == "image":
            lanes = original_image_lanes(lanes, sample["original_shapes"][0], cfg)
        _sync(device)
        elapsed = (time.perf_counter() - start) * 1000
        record = sample["records"][0]
        model_ms.append(model_elapsed)
        pipeline_ms.append(elapsed)
        pred = [lane["points"] for lane in lanes]
        row = {
            "id": record["id"],
            "lanes": lanes,
            "model_ms": model_elapsed,
            "pipeline_ms_excluding_io": elapsed,
            "sparsity": result["sparsity"],
        }
        if score:
            labels = sample["targets"]["instance"][0].numpy()
            probability = result["confidence"][0, 0].sigmoid().cpu().numpy()
            pixels += pixel_counts(probability >= cfg.decode.confidence, labels > 0, labels >= 0)
            if cfg.model.mode == "image":
                truth = record["lanes_2d"]
                c50 = lane_iou_counts(pred, truth, sample["original_shapes"][0], 0.5)
                c75 = lane_iou_counts(pred, truth, sample["original_shapes"][0], 0.75)
                if "native_lanes" in record:
                    native = sample_at_rows(pred, record["h_samples"])
                    ts = evaluate_frame(
                        native, record["native_lanes"], record["h_samples"], elapsed
                    )
                    tusimple.append(ts)
                    row["tusimple"] = {
                        "raw_file": record["id"],
                        "lanes": native,
                        "h_samples": record["h_samples"],
                        "run_time": elapsed,
                    }
            else:
                c50, cd = lane_3d_counts(pred, record["lanes_3d"], 0.5)
                c75 = c50
                distances.extend(cd)
            total50.add(c50)
            total75.add(c75)
            per_frame50.append(c50)
            per_frame75.append(c75)
            groups.append(record["group"])
            row["counts_50"] = {"tp": c50.tp, "fp": c50.fp, "fn": c50.fn}
        rows.append(row)
    write_jsonl(out / "predictions.jsonl", rows)
    report = {
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": state["epoch"],
        "evaluation_manifest_sha256": audit["manifest_sha256"],
        "evaluation_dataset_sha256": audit["dataset_sha256"],
        "split": split,
        "samples": len(rows),
        "mode": cfg.model.mode,
        "model_latency_ms": {
            "mean": float(np.mean(model_ms)),
            "p95": float(np.percentile(model_ms, 95)),
        },
        "pipeline_latency_ms_excluding_io": {
            "mean": float(np.mean(pipeline_ms)),
            "p95": float(np.percentile(pipeline_ms, 95)),
        },
        "timing_note": "Batch 1, CPU/CUDA synchronized; includes first forward; excludes file loading/preprocessing",
        "device": device,
        "reproduction_claim": False,
    }
    if score:
        report["pixel_diagnostic"] = pixel_metrics(pixels)
        report["lane_iou_50" if cfg.model.mode == "image" else "diagnostic_3d_at_0.5m"] = (
            total50.metrics()
        )
        report["group_bootstrap_f1_95ci"] = grouped_bootstrap_f1(per_frame50, groups)
        if cfg.model.mode == "image":
            report["lane_iou_75"] = total75.metrics()
            report["lane_metric_note"] = (
                "30px raster Hungarian matching; diagnostic, not official CULane evaluator"
            )
        else:
            report["matched_mean_chamfer_m"] = float(np.mean(distances)) if distances else None
            report["lane_metric_note"] = (
                "Symmetric arc-resampled Chamfer; not official ONCE protocol"
            )
        if tusimple:
            report["tusimple_protocol"] = {
                k: float(np.mean([v[k] for v in tusimple])) for k in tusimple[0]
            }
            write_jsonl(out / "tusimple-submission.jsonl", [r["tusimple"] for r in rows])
    write_json(out / "metrics.json", report)
    return report
