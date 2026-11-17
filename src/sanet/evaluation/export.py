"""Export diagnostic predictions to native CULane/ONCE evaluator layouts."""

from pathlib import Path

import numpy as np

from sanet.io import atomic_text, read_jsonl, rooted, write_json


def export_predictions(predictions, manifest, output_dir, format):
    records = {r["id"]: r for r in read_jsonl(manifest)}
    count = 0
    for prediction in read_jsonl(predictions):
        if prediction["id"] not in records:
            raise ValueError(f"Unknown prediction id {prediction['id']}")
        record = records[prediction["id"]]
        name = str(Path(record["id"]).with_suffix(".lines.txt" if format == "culane" else ".json"))
        destination = rooted(output_dir, name)
        if format == "culane":
            lines = []
            for lane in prediction["lanes"]:
                p = np.asarray(lane["points"], dtype=float)
                if p.ndim != 2 or p.shape[1] != 2:
                    raise ValueError("CULane export requires original-image 2D coordinates")
                lines.append(" ".join(f"{v:.3f}" for v in p.flatten()))
            atomic_text(destination, "\n".join(lines) + ("\n" if lines else ""))
        elif format == "once":
            view = record.get("views", [record])[0]
            transform = np.asarray(view["T_ego_from_camera"], dtype=float)
            lanes = []
            for lane in prediction["lanes"]:
                ego = np.asarray(lane["points"], dtype=float)
                if ego.ndim != 2 or ego.shape[1] != 3:
                    raise ValueError("ONCE export requires metric ego 3D coordinates")
                optical = (ego - transform[:3, 3]) @ transform[:3, :3]
                optical = optical[optical[:, 2] > 0]
                if len(optical) >= 2:
                    lanes.append({"points": optical.tolist(), "score": lane["score"]})
            write_json(destination, {"lanes": lanes})
        else:
            raise ValueError(f"Unknown export format {format}")
        count += 1
    return {"exported_frames": count, "format": format, "output": str(output_dir)}
