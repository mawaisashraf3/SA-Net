"""Separate pixel diagnostics, lane-instance matching, and diagnostic 3D distance."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def metrics(self):
        p = self.tp / max(1, self.tp + self.fp)
        r = self.tp / max(1, self.tp + self.fn)
        return {
            **asdict(self),
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
        }


def threshold_match(quality: np.ndarray, threshold: float):
    """Maximize valid match cardinality first, then quality (quality in [0,1])."""
    if not quality.size:
        return []
    valid = quality >= threshold
    utility = valid * (min(quality.shape) + 1 + quality)
    rows, cols = linear_sum_assignment(-utility)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]


def lane_masks(lanes, shape, width=30):
    masks = []
    for lane in lanes:
        mask = np.zeros(shape, dtype=np.uint8)
        if len(lane) >= 2:
            cv2.polylines(mask, [np.rint(lane).astype(np.int32)], False, 1, thickness=width)
        masks.append(mask.astype(bool))
    return masks


def lane_iou_counts(prediction, truth, shape, threshold=0.5, width=30):
    pm, tm = lane_masks(prediction, shape, width), lane_masks(truth, shape, width)
    quality = np.zeros((len(pm), len(tm)))
    for i, p in enumerate(pm):
        for j, t in enumerate(tm):
            union = np.logical_or(p, t).sum()
            quality[i, j] = np.logical_and(p, t).sum() / union if union else 0
    matches = threshold_match(quality, threshold)
    return Counts(len(matches), len(pm) - len(matches), len(tm) - len(matches))


def resample_curve(points, spacing=0.5):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or not len(p):
        raise ValueError("Cannot resample empty or malformed curve")
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-8]
    distance, p = distance[keep], p[keep]
    if len(p) == 1:
        return p
    samples = np.linspace(0, distance[-1], max(2, int(np.ceil(distance[-1] / spacing)) + 1))
    return np.column_stack([np.interp(samples, distance, p[:, i]) for i in range(p.shape[1])])


def chamfer(a, b):
    """Mean of both directed nearest-neighbor Euclidean distances, in input units."""
    distance = cdist(resample_curve(a), resample_curve(b))
    return float((distance.min(0).mean() + distance.min(1).mean()) / 2)


def lane_3d_counts(prediction, truth, threshold=0.5):
    distances = np.array([[chamfer(p, t) for t in truth] for p in prediction], dtype=float)
    distances = distances.reshape(len(prediction), len(truth))
    # Monotone transform into [0,1] for cardinality matching.
    quality = np.exp(-distances)
    matches = threshold_match(quality, np.exp(-threshold))
    return (
        Counts(len(matches), len(prediction) - len(matches), len(truth) - len(matches)),
        [float(distances[i, j]) for i, j in matches],
    )


def pixel_counts(pred, truth, valid):
    p, t, v = np.asarray(pred, bool), np.asarray(truth, bool), np.asarray(valid, bool)
    return np.array(
        [np.sum(p & t & v), np.sum(p & ~t & v), np.sum(~p & t & v), np.sum(~p & ~t & v)],
        dtype=np.int64,
    )


def pixel_metrics(counts):
    tp, fp, fn, tn = map(int, counts)
    result = Counts(tp, fp, fn).metrics()
    result.update(
        tn=tn, iou=tp / max(1, tp + fp + fn), accuracy=(tp + tn) / max(1, tp + fp + fn + tn)
    )
    return result


def bootstrap_f1(per_frame: list[Counts], iterations=2000, seed=17):
    """Frame bootstrap diagnostic; use grouped_bootstrap_f1 for temporally related frames."""
    return grouped_bootstrap_f1(
        per_frame, [str(i) for i in range(len(per_frame))], iterations, seed
    )


def grouped_bootstrap_f1(per_frame, groups, iterations=2000, seed=17):
    if not per_frame or len(per_frame) != len(groups):
        raise ValueError("Need a group for every frame")
    unique = sorted(set(groups))
    if len(unique) < 2:
        return {
            "low": None,
            "high": None,
            "groups": len(unique),
            "reason": "At least two independent groups required",
        }
    totals = []
    for group in unique:
        c = Counts()
        for frame, frame_group in zip(per_frame, groups):
            if frame_group == group:
                c.add(frame)
        totals.append([c.tp, c.fp, c.fn])
    array, rng, values = np.array(totals), np.random.default_rng(seed), []
    for _ in range(iterations):
        tp, fp, fn = array[rng.integers(len(unique), size=len(unique))].sum(0)
        values.append(2 * tp / max(1, 2 * tp + fp + fn))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "groups": len(unique),
        "iterations": iterations,
        "method": "group percentile bootstrap",
    }
