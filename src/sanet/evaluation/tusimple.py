"""Clean implementation of TuSimple point accuracy/FP/FN protocol.

Protocol reference: TuSimple/tusimple-benchmark evaluate/lane.py (Apache-2.0).
Uses numpy least-squares instead of sklearn regression; retains matching and
special-case semantics, including possible multiple GT matches to one prediction.
"""

import numpy as np


def evaluate_frame(prediction, truth, h_samples, runtime_ms=0):
    if any(len(lane) != len(h_samples) for lane in prediction + truth):
        raise ValueError("TuSimple lane length must match h_samples")
    if not truth:
        raise ValueError("Official TuSimple protocol expects at least one annotated lane")
    if runtime_ms > 200 or len(truth) + 2 < len(prediction):
        return {"accuracy": 0.0, "fp_rate": 0.0, "fn_rate": 1.0}
    scores, matched, missed = [], 0, 0
    ys = np.asarray(h_samples)
    for target in truth:
        t = np.asarray(target)
        valid = t >= 0
        slope = np.polyfit(ys[valid], t[valid], 1)[0] if valid.sum() >= 2 else 0.0
        tolerance = 20 / np.cos(np.arctan(slope))
        t = np.where(t < 0, -100, t)
        accuracy = []
        for lane in prediction:
            p = np.asarray(lane)
            p = np.where(p < 0, -100, p)
            accuracy.append(float((np.abs(p - t) < tolerance).mean()))
        best = max(accuracy, default=0.0)
        scores.append(best)
        if best < 0.85:
            missed += 1
        else:
            matched += 1
    fp = len(prediction) - matched
    if len(truth) > 4 and missed > 0:
        missed -= 1
    total = sum(scores) - (min(scores) if len(truth) > 4 else 0)
    return {
        "accuracy": total / max(1, min(4, len(truth))),
        "fp_rate": fp / len(prediction) if prediction else 0.0,
        "fn_rate": missed / max(1, min(len(truth), 4)),
    }


def sample_at_rows(lanes, rows):
    result = []
    for lane in lanes:
        p = np.asarray(lane, dtype=float)
        order = np.argsort(p[:, 1], kind="stable")
        p = p[order]
        y, index = np.unique(p[:, 1], return_index=True)
        p = p[index]
        if len(p) < 2:
            continue
        xs = np.interp(rows, y, p[:, 0])
        result.append(
            [int(round(x)) if y[0] <= row <= y[-1] and x >= 0 else -2 for x, row in zip(xs, rows)]
        )
    return result
