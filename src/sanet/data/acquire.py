"""Pinned real TuSimple subset. Original usage terms apply; images stay local."""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from sanet.data.adapters import tusimple
from sanet.io import rooted, sha256, write_json, write_jsonl

MIRROR = "dotblue/TuSimple"
REVISION = "3ab1eb751e3036a83b1e40c33a1da7e77b240c84"
BASE = f"https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/"


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                content = response.read()
            if not content:
                raise ValueError(f"Empty response from {url}")
            temporary = destination.with_suffix(destination.suffix + ".download")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return {
                "path": str(destination),
                "url": url,
                "bytes": len(content),
                "sha256": sha256(destination),
            }
        except (OSError, ValueError) as e:
            last_error = e
    raise RuntimeError(f"Download failed: {url}: {last_error}")


def acquire_small(root, train_count=24, val_count=8, test_count=8):
    """Disjoint capture dates from native training labels; not official test data."""
    root = Path(root)
    if min(train_count, val_count, test_count) < 1:
        raise ValueError("Each count must be positive")
    files, records, availability = [], [], {}
    metadata_path = root / "mirror-metadata.json"
    files.append(
        fetch(f"https://huggingface.co/api/datasets/{MIRROR}/revision/{REVISION}", metadata_path)
    )
    metadata = json.loads(metadata_path.read_text())
    if metadata["sha"] != REVISION:
        raise ValueError("Mirror revision mismatch")
    available = {item["rfilename"] for item in metadata["siblings"]}
    for date, split, count in [
        ("0313", "train", train_count),
        ("0531", "val", val_count),
        ("0601", "test", test_count),
    ]:
        labels = root / f"label_data_{date}.json"
        files.append(fetch(BASE + labels.name, labels))
        native_rows = tusimple(labels, split)
        indexed = [(i, r) for i, r in enumerate(native_rows) if r["image"] in available]
        availability[date] = {
            "native_annotations": len(native_rows),
            "mirror_images_available": len(indexed),
        }
        rows = [r for _, r in indexed]
        if count > len(rows):
            raise ValueError(f"Requested {count} {split} images; only {len(rows)} available")
        indices = [round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)]
        for index in indices:
            record = rows[index]
            record["group"] = f"capture-date-{date}"
            record["source_annotation_index"] = indexed[index][0]
            records.append(record)

    def download(record):
        return fetch(BASE + quote(record["image"]), rooted(root, record["image"]))

    with ThreadPoolExecutor(max_workers=4) as executor:
        files.extend(executor.map(download, records))
    for file in files:
        file["path"] = str(
            Path(file["path"]).relative_to(
                root.resolve() if Path(file["path"]).is_absolute() else root
            )
        )
    write_jsonl(root / "manifest.jsonl", records)
    write_json(
        root / "provenance.json",
        {
            "mirror": MIRROR,
            "revision": REVISION,
            "purpose": "limited real-data engineering experiment; not benchmark reproduction",
            "partition": "disjoint capture dates from native training annotations",
            "counts": {"train": train_count, "val": val_count, "test": test_count},
            "mirror_card_license": "apache-2.0",
            "original_terms_verified": False,
            "availability": availability,
            "files": files,
        },
    )
    return {"manifest": str(root / "manifest.jsonl"), "images": len(records), "revision": REVISION}
