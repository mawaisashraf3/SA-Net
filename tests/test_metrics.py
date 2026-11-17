import numpy as np
import pytest
import torch

from sanet.evaluation.decode import decode, original_image_lanes
from sanet.evaluation.metrics import (
    Counts,
    chamfer,
    grouped_bootstrap_f1,
    lane_3d_counts,
    lane_iou_counts,
    pixel_counts,
    pixel_metrics,
    threshold_match,
)
from sanet.evaluation.tusimple import evaluate_frame, sample_at_rows


def test_matching_cardinality_before_quality():
    quality = np.array([[0.99, 0.51], [0.51, 0.49]])
    assert set(threshold_match(quality, 0.5)) == {(0, 1), (1, 0)}
    assert threshold_match(np.empty((0, 3)), 0.5) == []


def test_iou_empty_and_duplicate_predictions():
    lane = [[10, 2], [10, 30]]
    assert lane_iou_counts([lane], [lane], (40, 40)).metrics()["f1"] == 1
    assert lane_iou_counts([lane, lane], [lane], (40, 40)).fp == 1
    assert lane_iou_counts([], [lane], (40, 40)).fn == 1
    assert lane_iou_counts([], [], (40, 40)).tp == 0


def test_chamfer_resampling_and_3d_match():
    a, b = [[0, 0, 0], [10, 0, 0]], [[0, 1, 0], [10, 1, 0]]
    assert chamfer(a, a) == 0
    assert chamfer(a, b) == pytest.approx(1)
    counts, cd = lane_3d_counts([a, b], [a])
    assert counts.tp == 1 and counts.fp == 1 and cd == [0]


def test_pixels_and_bootstrap():
    counts = pixel_counts([1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1])
    assert pixel_metrics(counts)["iou"] == pytest.approx(1 / 3)
    assert grouped_bootstrap_f1([Counts(1, 0, 0)], ["x"])["low"] is None
    ci = grouped_bootstrap_f1([Counts(1, 0, 0), Counts(1, 0, 0)], ["a", "b"], 20)
    assert ci["low"] == ci["high"] == 1


def test_tusimple_special_cases():
    truth = [[10, 11, 12], [50, 51, 52]]
    perfect = evaluate_frame(truth, truth, [1, 2, 3])
    assert perfect == {"accuracy": 1.0, "fp_rate": 0.0, "fn_rate": 0.0}
    assert evaluate_frame(truth, truth, [1, 2, 3], 201)["fn_rate"] == 1
    assert evaluate_frame([], truth, [1, 2, 3])["accuracy"] == 0
    with pytest.raises(ValueError):
        evaluate_frame([[1, 2]], truth, [1, 2, 3])
    five = [[i * 100, i * 100 + 1, i * 100 + 2] for i in range(5)]
    assert evaluate_frame(five[:-1], five, [1, 2, 3])["fn_rate"] == 0


def test_decoder_geometry_and_tusimple_export(config):
    config.decode.confidence = 0.8
    h, w = 32, 64
    confidence = torch.full((1, 1, h, w), -20.0)
    confidence[0, 0, :, 8] = 20
    output = {
        "confidence": confidence,
        "embedding": torch.zeros(1, 2, h, w),
        "offset": torch.zeros(1, 1, h, w),
        "height": torch.zeros(1, 1, h, w),
    }
    lanes = decode(output, config)[0]
    assert len(lanes) == 1 and len(lanes[0]["points"]) == 32
    original = original_image_lanes(lanes, (64, 128), config)
    assert original[0]["points"][0] == [16.5, 0.5]
    rows = sample_at_rows([original[0]["points"]], [0, 1, 30, 63])
    assert rows[0][0] == rows[0][-1] == -2


def test_native_exports(config, records, tmp_path):
    import json

    from sanet.evaluation.export import export_predictions
    from sanet.io import write_jsonl

    p = tmp_path / "predictions.jsonl"
    write_jsonl(
        p, [{"id": "0", "lanes": [{"points": [[2.0, -2, 0], [10.0, -2, 0]], "score": 0.9}]}]
    )
    export_predictions(p, config.data.manifest, tmp_path / "once-export", "once")
    native = json.loads((tmp_path / "once-export/0.json").read_text())["lanes"][0]["points"]
    np.testing.assert_allclose(native, [[2, 1.5, 2], [2, 1.5, 10]])
    write_jsonl(p, [{"id": "0", "lanes": [{"points": [[2.0, 4], [8.0, 10]], "score": 0.9}]}])
    export_predictions(p, config.data.manifest, tmp_path / "culane-export", "culane")
    assert (tmp_path / "culane-export/0.lines.txt").read_text().startswith("2.000 4.000")
