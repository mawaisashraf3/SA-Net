"""Canonical image/sensor dataset, calibration-aware crop/resize/flip and batching."""

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from sanet.config import Config
from sanet.data.targets import bev_targets, rasterize
from sanet.geometry import project, resize_matrix, validate_calibration
from sanet.io import read_jsonl, rooted


class LaneDataset(Dataset):
    def __init__(self, cfg: Config, split: str, augment: bool = False):
        self.cfg = cfg
        self.records = [r for r in read_jsonl(cfg.data.manifest) if r["split"] == split]
        self.augment = augment
        if not self.records:
            raise ValueError(f"No {split} samples in {cfg.data.manifest}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record, cfg = self.records[index], self.cfg
        d = cfg.data
        images, intrinsics, transforms, image_instances, depth_labels = [], [], [], [], []
        views = record.get("views", [record])
        flip = self.augment and torch.rand(()).item() < d.flip_probability
        clouds = (
            np.load(rooted(d.root, record["points"]), allow_pickle=False)
            if record.get("points")
            else np.empty((0, 4), dtype=np.float32)
        )
        points = torch.as_tensor(clouds, dtype=torch.float32)
        points = points[torch.isfinite(points).all(1)]
        # Annotated metric keypoints are targets only, never model inputs.
        labelled_points = [
            torch.tensor(lane, dtype=torch.float32) for lane in record.get("lanes_3d", [])
        ]
        depth_points = torch.cat([points[:, :3], *labelled_points])
        original_shape = None
        for view in views:
            with Image.open(rooted(d.root, view["image"])) as source:
                im = source.convert("RGB")
            ow, oh = im.size
            original_shape = (oh, ow)
            affine = resize_matrix(oh, ow, d.height, d.width, d.crop_top)
            im = im.crop((0, d.crop_top, ow, oh)).resize(
                (d.width, d.height), Image.Resampling.BILINEAR
            )
            if flip:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                reflection = torch.tensor([[-1.0, 0, d.width - 1], [0, 1, 0], [0, 0, 1]])
                affine = reflection @ affine
            if self.augment:
                brightness = 1 + (2 * torch.rand(()).item() - 1) * d.brightness
                contrast = 1 + (2 * torch.rand(()).item() - 1) * d.contrast
                im = ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(brightness)).enhance(
                    contrast
                )
            rgb = torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float() / 255
            images.append(
                (rgb - rgb.new_tensor([0.485, 0.456, 0.406])[:, None, None])
                / rgb.new_tensor([0.229, 0.224, 0.225])[:, None, None]
            )
            lanes = []
            for lane in view.get("lanes_2d", []):
                p = torch.tensor(lane, dtype=torch.float32)
                q = torch.cat([p, torch.ones_like(p[:, :1])], 1) @ affine.T
                lanes.append(q[:, :2].tolist())
            thickness = max(1, round(d.line_width * d.width / ow))
            image_instances.append(rasterize(lanes, d.height, d.width, thickness))
            depth = torch.full((d.height, d.width), -1.0)
            if cfg.model.mode == "bev":
                k = affine @ torch.tensor(view["intrinsics"], dtype=torch.float32)
                t = torch.tensor(view["T_ego_from_camera"], dtype=torch.float32)
                validate_calibration(k, t)
                intrinsics.append(k)
                transforms.append(t)
                if len(depth_points):
                    uv, z = project(depth_points, k, t)
                    px = uv.round().long()
                    valid = (
                        (z > 0)
                        & torch.isfinite(uv).all(1)
                        & (px[:, 0] >= 0)
                        & (px[:, 0] < d.width)
                        & (px[:, 1] >= 0)
                        & (px[:, 1] < d.height)
                    )
                    # Nearest visible depth at pixel; no invented dense depth supervision.
                    flat = depth.flatten().fill_(torch.inf)
                    indices = px[valid, 1] * d.width + px[valid, 0]
                    flat.scatter_reduce_(0, indices, z[valid], reduce="amin", include_self=True)
                    depth = torch.where(torch.isfinite(depth), depth, -1.0)
            depth_labels.append(depth)
        image_instances = torch.stack(image_instances)
        if cfg.model.mode == "bev":
            targets = bev_targets(record.get("lanes_3d", []), cfg.model.grid)
        else:
            inst = image_instances[0]
            zeros = torch.zeros((1, d.height, d.width))
            targets = {
                "instance": inst,
                "offset": zeros,
                "height": zeros.clone(),
                "height_valid": torch.zeros_like(zeros, dtype=torch.bool),
            }
        targets.update(image_instance=image_instances, depth=torch.stack(depth_labels))
        return {
            "images": torch.stack(images),
            "points": points,
            "intrinsics": torch.stack(intrinsics) if intrinsics else torch.empty((0, 3, 3)),
            "transforms": torch.stack(transforms) if transforms else torch.empty((0, 4, 4)),
            "targets": targets,
            "id": record["id"],
            "record": record,
            "original_shape": original_shape,
        }


def collate(samples: list[dict]) -> dict:
    if len({s["images"].shape for s in samples}) != 1:
        raise ValueError("A batch must have a common number of cameras and image shape")
    return {
        "images": torch.stack([s["images"] for s in samples]),
        "points": [s["points"] for s in samples],
        "intrinsics": torch.stack([s["intrinsics"] for s in samples]),
        "transforms": torch.stack([s["transforms"] for s in samples]),
        "targets": {
            k: torch.stack([s["targets"][k] for s in samples]) for k in samples[0]["targets"]
        },
        "ids": [s["id"] for s in samples],
        "records": [s["record"] for s in samples],
        "original_shapes": [s["original_shape"] for s in samples],
    }


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    return batch
