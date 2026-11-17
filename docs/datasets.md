# Datasets, acquisition and integrity

Dataset assets are external, immutable inputs; they are not included in the code license or committed to Git. The attached manuscript does not provide actual data, weights, calibration files or split manifests. Download from the authoritative sources and comply with their current terms. Source links below were inspected during development on 2026-11-17 (client-provided date).

| Dataset | Verified source and format | Availability / terms | Supported path |
|---|---|---|---|
| CULane | [Official site](https://xingangpan.github.io/projects/CULane.html), [SCNN](https://github.com/XingangPan/SCNN). 88,880 train / 9,675 val / 34,680 test, 1640×590. Images, per-frame `.lines.txt` xy polylines, split lists, segmentation labels. | Public download links; academic research use under official site terms. The site notes corrected train/val raw annotations after 2018-04-16. Obtain the corrected release. | Native converter, image-plane training, native prediction export. Approximate 30px raster lane IoU diagnostics; official SCNN evaluation remains required for benchmark claims. |
| TuSimple | [Official benchmark](https://github.com/TuSimple/tusimple-benchmark/blob/master/doc/lane_detection/readme.md). 3,626 training annotated last frames, 2,782 test clips; 1280×720. JSONL `raw_file`, `h_samples`, `lanes`; negative x = absent point. | Public research benchmark; official tooling repository Apache-2.0 does not establish rights for every dataset asset. The manuscript's 6.4K *training* table entry confuses the total. 358 validation frames are a partition of the native training data, not extra observations. | Native converter, point accuracy/FP/FN protocol, native submission JSONL. |
| BDD100K | [Official site](https://www.bdd100k.com/), [toolkit](https://github.com/bdd100k/bdd100k). Common image split 70K/10K/20K, task label availability varies. Frame arrays with lane `poly2d` and attributes; 1280×720. | Follow official access/download and license agreements; do not infer task-specific label counts from image counts. | Converter for legacy/Scalabel categories `lane` and `lane/*`, L vertices and cubic `CCC` runs. Other taxonomy versions need explicit mapping. Do not silently interpret unrelated objects as lane markings. |
| FoggyLane | Manuscript reference [24], R. Zhang et al., “A Robust Real-Time Lane Detection Method with Fog-Enhanced Feature Fusion for Foggy Conditions,” 2025. | Citation lacks DOI/repository/access URL. A unique authoritative dataset/release could not be verified. About 3.5K total conflicts with 3.3K+0.35K+2.8K. License and split identities unresolved. | Schedule preset and canonical 2D schema, pending verified data. No dedicated native converter or acquisition claim. Synthetic fog is not substituted. |
| ONCE-3DLanes | [Official details](https://once-3dlanes.github.io/3dlanes/), [benchmark](https://github.com/once-3dlanes/once_3dlanes_benchmark). About 200K/3K/8K, 1920×1020; raw camera-space lane xyz and 3×4 `calibration` intrinsics. Images obtained through underlying ONCE release. | Labels public download; original ONCE image/point-cloud terms apply separately. Hidden test annotations may be unavailable. Native lane release is monocular; raw LiDAR synchronization/extrinsics are not automatically supplied by 3D lane labels. | Native annotation converter **requires** a declared optical-camera→ego transform; calibrated camera-only BEV or genuine LiDAR fusion; native ONCE export. Official evaluator is external. |

The ONCE descriptive axes sentence conflicts with its example values, intrinsic matrix and evaluator use of the third coordinate as longitudinal depth. The adapter follows optical camera xyz and requires a user-provided rigid transform. Reproject sample labels onto real images before preparing a release; do not guess sensor extrinsics from nominal camera height.

## Real subset used here

`sanet acquire-small --root data/tusimple-small` downloads **24 train + 8 validation + 8 test** real images and their existing annotations from [dotblue/TuSimple](https://huggingface.co/datasets/dotblue/TuSimple), pinned at `3ab1eb751e3036a83b1e40c33a1da7e77b240c84`. The mirror's card advertises Apache-2.0; this is a mirror assertion, not independent verification of the original image terms. No images are redistributed here.

Native annotation files contain 2,858 / 358 / 410 records for dates 0313 / 0531 / 0601. The mirror includes matching images for only 200 / 34 / 33 of them. Selection intersects available files with annotations, then takes evenly spaced indices, without creating labels or images. Entire dates are assigned to separate partitions. `provenance.json` records available counts, original annotation indices, pinned URLs, file sizes and SHA-256 hashes.

All three partitions come from the original **training** release. The local `test` partition is an engineering holdout, not TuSimple's official 2,782-image test set. It has one capture-date group; a group bootstrap confidence interval is therefore withheld. The initial holdout was viewed during development; subsequent replay is a reproducibility check, not a new independent holdout experiment. Broader accuracy/generalization claims require new untouched sequences and repeated seeds.

## Canonical JSONL schema

Image-only training record (schematic; coordinates below illustrate the schema):

```json
{"id":"clips/date/clip/20.jpg","group":"capture-session","split":"train","image":"clips/date/clip/20.jpg","lanes_2d":[[[120,300],[80,700]]],"source":"TuSimple"}
```

`split`: `train`, `val`, `test`, or `inference`. All paths are relative to `--root`; parent traversal and external absolute paths are rejected. Empty `lanes_2d: []` means an annotated negative frame; omitting it means missing annotation and is rejected for training/evaluation. `group` identifies the recording sequence/session, not a random frame ID. Audit rejects a group present in different partitions. CULane defaults to image-parent sequence paths; review these against the official split and use independently justified finer sequence IDs if needed. Do not evade the check by assigning random frame groups.

BEV records additionally require:

* `lanes_3d`: list of Nx3 polylines in ego meters; x forward, y left, z up.
* `intrinsics`: 3×3 matrix at the original image resolution.
* `T_ego_from_camera`: proper rigid 4×4 transform; optical camera x right, y down, z forward.
* `points`: optional relative `.npy`, float Nx4 `[x,y,z,intensity]` in the same ego frame. Missing LiDAR remains an empty tensor, with no pseudo-points.

Multiple cameras use `views: [{image, intrinsics, T_ego_from_camera, lanes_2d}, ...]`. Each view has its own segmentation annotation; the number of views must be constant within a batch. Top-level metric annotations and point cloud are shared. Image-plane mode uses exactly one camera. An inference record may omit all labels. In inference, labels present in records are ignored by the model; target construction is separate from sensor inputs.

## Conversion and audit

```bash
sanet convert tusimple --labels data/tusimple/label_data_0313.json --split train --output data/tusimple/train.jsonl
sanet convert culane --root data/culane --labels data/culane/list/train.txt --split train --output data/culane/train.jsonl
sanet convert bdd100k --labels data/bdd100k/labels/lane_train.json --image-prefix images/train --split train --output data/bdd100k/train.jsonl
sanet convert once --root data/once/annotations --labels data/once/train.txt --image-prefix images --transform camera-to-ego.json --split train --output data/once/train.jsonl
```

Convert validation/test separately, concatenate into one manifest (e.g. `cat train.jsonl val.jsonl test.jsonl > manifest.jsonl`), then audit:

```bash
sanet audit --manifest data/tusimple-small/manifest.jsonl --root data/tusimple-small --output runs/data-audit.json
# Calibrated data:
sanet audit --manifest data/once/manifest.jsonl --root data/once --mode bev
```

The audit checks IDs, sequence overlap, image readability and duplicate image bytes, finite labels, point-cloud structure, cross-partition duplicate point bytes, matrix shapes, finite/invertible intrinsics and proper rigid transforms. It hashes manifest **and assets** for checkpoint recovery checks. It does not prove temporal independence, validate sensor timestamps or detect visually near-duplicate frames; data curators must verify those aspects.

No imputation of labels, sensor geometry, depth, or classes is performed. Invalid assets fail early. Empty target masks are supported. Collision cells are ignored; positives are weighted with a capped training-only foreground/background ratio. Masks use nearest/discrete operations; RGB uses bilinear resize. Crop/resize/flip matrices transform both lane coordinates and intrinsics. Brightness/contrast affect RGB only. Arbitrary affine augmentation from the manuscript is omitted because the exact distribution was not provided.
