import json

import numpy as np
import pytest
import torch

from sanet.config import from_dict
from sanet.data import adapters
from sanet.data.audit import audit_manifest
from sanet.data.dataset import LaneDataset, collate
from sanet.data.targets import bev_targets, rasterize
from sanet.io import rooted, write_jsonl


def test_audit_and_batch(config, records):
    report = audit_manifest(config.data.manifest, config.data.root)
    assert report["splits"] == {"train": 2, "val": 1, "test": 1}
    dataset = LaneDataset(config, "train")
    batch = collate([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 1, 3, 32, 64)
    assert set(torch.unique(batch["targets"]["instance"]).tolist()) == {0, 1, 2}


@pytest.mark.parametrize("kind", ["id", "group", "bytes", "label", "calibration"])
def test_audit_rejects_invalid(config, records, kind):
    if kind == "id":
        records[2]["id"] = records[0]["id"]
    elif kind == "group":
        records[2]["group"] = records[0]["group"]
    elif kind == "bytes":
        records[2]["image"] = records[0]["image"]
    elif kind == "label":
        records[2]["lanes_2d"][0][0][0] = float("inf")
    else:
        config.model.mode = "bev"
        del records[2]["intrinsics"]
    # Nonfinite fixture is intentionally invalid JSON numeric data: write manually.
    open(config.data.manifest, "w").write("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError):
        audit_manifest(config.data.manifest, config.data.root, config.model.mode)


def test_bev_targets_and_flip_geometry(config, records):
    config.model.mode = "bev"
    config.data.flip_probability = 1.0
    a = LaneDataset(config, "train", False)[0]
    b = LaneDataset(config, "train", True)[0]
    assert a["targets"]["instance"].shape == (16, 16)
    torch.testing.assert_close(a["targets"]["offset"], b["targets"]["offset"])
    torch.testing.assert_close(
        b["intrinsics"][0, 0], torch.tensor([-1.0, 0, 63.0]) @ a["intrinsics"][0]
    )
    assert (a["targets"]["depth"] > 0).any()


def test_collision_is_ignored(config):
    lane = [[1, -2, 0], [10, -2, 0]]
    target = bev_targets([lane, lane], config.model.grid)
    assert (target["instance"] == -1).any()
    assert not target["height_valid"].any()
    raster = rasterize([[[2, 0], [2, 10]], [[2, 0], [2, 10]]], 12, 12, 1)
    assert (raster == -1).sum() == 11


def test_tusimple_adapter_lengths(tmp_path):
    p = tmp_path / "labels.jsonl"
    write_jsonl(
        p, [{"raw_file": "clips/date/clip/20.jpg", "h_samples": [1, 2, 3], "lanes": [[-2, 10, 20]]}]
    )
    row = adapters.tusimple(p, "train")[0]
    assert row["lanes_2d"] == [[[10.0, 2.0], [20.0, 3.0]]]
    write_jsonl(p, [{"raw_file": "x", "h_samples": [1, 2], "lanes": [[1]]}])
    with pytest.raises(ValueError):
        adapters.tusimple(p, "train")


def test_native_culane_bdd_once(tmp_path):
    (tmp_path / "lane.lines.txt").write_text("1 2 3 4\n")
    (tmp_path / "list.txt").write_text("/lane.jpg /mask.png 1 1 0 0\n")
    assert adapters.culane(tmp_path, tmp_path / "list.txt", "train")[0]["lanes_2d"] == [
        [[1.0, 2.0], [3.0, 4.0]]
    ]
    bdd = [
        {
            "name": "frame.jpg",
            "labels": [
                {
                    "category": "lane/single white",
                    "poly2d": [{"vertices": [[0, 0], [1, 0], [2, 1], [3, 1]], "types": "LCCC"}],
                }
            ],
        }
    ]
    (tmp_path / "bdd.json").write_text(json.dumps(bdd))
    p = adapters.bdd100k(tmp_path / "bdd.json", "val")[0]["lanes_2d"][0]
    np.testing.assert_allclose(p[-1], [3, 1])
    (tmp_path / "seq").mkdir()
    (tmp_path / "seq/frame.json").write_text(
        json.dumps(
            {
                "calibration": [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0]],
                "lanes": [[[1, 2, 4], [2, 2, 8]]],
            }
        )
    )
    (tmp_path / "once.txt").write_text("seq/frame.jpg\n")
    row = adapters.once(tmp_path, tmp_path / "once.txt", "train", "images", torch.eye(4).tolist())[
        0
    ]
    assert row["lanes_2d"] == [[[0.5, 1.0], [0.5, 0.5]]]


def test_path_and_config_guards(tmp_path):
    with pytest.raises(ValueError):
        rooted(tmp_path, "../escape")
    with pytest.raises(ValueError):
        from_dict({"model": {"unknown": True}})
    with pytest.raises(ValueError):
        from_dict({"model": {"top_k": 99}})
    with pytest.raises(ValueError):
        from_dict({"model": {"grid": {"resolution": 0.7}}})


def test_point_cloud_leakage_detected(config, records):
    records[2]["points"] = records[0]["points"]
    write_jsonl(config.data.manifest, records)
    with pytest.raises(ValueError, match="Point-cloud bytes"):
        audit_manifest(config.data.manifest, config.data.root)


def test_inference_does_not_require_targets(config, records):
    record = records[0]
    record["split"] = "inference"
    for key in ["lanes_2d", "lanes_3d"]:
        del record[key]
    write_jsonl(config.data.manifest, [record])
    audit_manifest(config.data.manifest, config.data.root)
    sample = LaneDataset(config, "inference")[0]
    assert not sample["targets"]["instance"].any()
