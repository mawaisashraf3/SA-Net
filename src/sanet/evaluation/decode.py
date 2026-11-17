"""Deterministic bounded embedding clustering and lane keypoint reconstruction."""

import numpy as np
import torch

from sanet.config import Config


def decode(output, cfg: Config):
    lanes_per_frame = []
    for b in range(len(output["confidence"])):
        confidence = output["confidence"][b, 0].detach().float().cpu()
        embeddings = output["embedding"][b].detach().float().cpu().permute(1, 2, 0)
        rows, columns = torch.where(confidence >= cfg.decode.confidence)
        scores = confidence[rows, columns]
        order = torch.argsort(scores, descending=True, stable=True)[: cfg.decode.max_candidates]
        rows, columns, scores = rows[order], columns[order], scores[order]
        vectors = embeddings[rows, columns]
        assigned = torch.zeros(len(rows), dtype=torch.bool)
        lanes = []
        for _ in range(cfg.decode.max_lanes):
            remaining = torch.where(~assigned)[0]
            if not len(remaining):
                break
            center = vectors[remaining[0]]
            # Three mean-shift refinement steps, then disjoint assignment.
            for _ in range(3):
                member = (~assigned) & (
                    torch.linalg.vector_norm(vectors - center, dim=1) < cfg.decode.embedding_radius
                )
                if member.any():
                    center = vectors[member].mean(0)
            member = (~assigned) & (
                torch.linalg.vector_norm(vectors - center, dim=1) < cfg.decode.embedding_radius
            )
            if not member.any():
                assigned[remaining[0]] = True
                continue
            assigned |= member
            r, c, s = rows[member], columns[member], scores[member]
            points = []
            for row in torch.unique(r, sorted=True)[:: cfg.decode.row_step]:
                good = r == row
                # Highest confidence cell per row avoids averaging separate peaks.
                j = torch.argmax(s[good])
                col = int(c[good][j])
                ri = int(row)
                if cfg.model.mode == "bev":
                    g = cfg.model.grid
                    offset = float(output["offset"][b, 0, ri, col].detach().cpu())
                    offset = float(np.clip(offset, -g.resolution / 2, g.resolution / 2))
                    height = float(output["height"][b, 0, ri, col].detach().cpu())
                    points.append(
                        [
                            g.x_min + (ri + 0.5) * g.resolution,
                            g.y_min + (col + 0.5) * g.resolution + offset,
                            height,
                        ]
                    )
                else:
                    points.append([float(col), float(ri)])
            if len(points) >= cfg.decode.min_points:
                lanes.append({"points": points, "score": float(s.mean())})
        lanes_per_frame.append(lanes)
    return lanes_per_frame


def original_image_lanes(lanes, original_shape, cfg):
    oh, ow = original_shape
    sy = (oh - cfg.data.crop_top) / cfg.data.height
    sx = ow / cfg.data.width
    return [
        {
            "points": [
                [(x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5 + cfg.data.crop_top]
                for x, y in lane["points"]
            ],
            "score": lane["score"],
        }
        for lane in lanes
    ]
