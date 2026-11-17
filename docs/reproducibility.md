# Installation, execution and reproducibility

Python >=3.10. The executed environment is recorded in [results/cpu-small/environment.json](../results/cpu-small/environment.json); exact tested direct package versions are in [constraints-tested.txt](../constraints-tested.txt). CPU is sufficient for unit/integration tests and the small real-data experiment. Full 320×800, batch32 ResNet-34 training generally requires a suitable GPU and external data.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# For CPU, install matching CPU wheels first:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m sanet --help
pytest -q
ruff check src tests
ruff format --check src tests
```

For an exact direct-dependency replay, use `pip install -c constraints-tested.txt -e '.[dev]'` after selecting the matching CPU wheel index. The constraints record the environment actually available here, not a promise that older Python/OS combinations have those exact wheels. CUDA users must select mutually compatible torch/torchvision/CUDA distributions and record their own environment.

No weights, data or third-party repositories are downloaded by importing the package or running its tests. `acquire-small` and `model.pretrained: true` are explicit network operations. Dataset assets and run checkpoints are excluded from Git. The explicitly distributed [compact engineering checkpoint](../artifacts/MODEL_CARD.md) is a small, measured exception; it has zero recorded lane-instance F1 and is intended for pipeline inspection. An installed wheel includes Python modules; clone the repository for configs, docs and results.

## Replay the limited real-data experiment

```bash
sanet acquire-small --root data/tusimple-small
sanet audit --manifest data/tusimple-small/manifest.jsonl --root data/tusimple-small --output runs/data-audit.json
sanet train --config configs/cpu-small.yaml --output runs/cpu-small
sanet evaluate --checkpoint runs/cpu-small/best.pt --manifest data/tusimple-small/manifest.jsonl --root data/tusimple-small --split test --output runs/cpu-small/evaluation
sanet report --run runs/cpu-small --evaluation runs/cpu-small/evaluation --output runs/cpu-small/report
```

Use a fresh output directory per new experiment. Existing checkpoints are protected against accidental overwrite. All paths in config files are resolved from the current working directory; run commands at repository root or use explicit absolute paths.

Recover a deliberately interrupted run without changing the original 12-epoch cosine schedule:

```bash
sanet train --config configs/cpu-small.yaml --output runs/recovery --stop-after 4
sanet train --config configs/cpu-small.yaml --output runs/recovery --resume runs/recovery/last.pt
```

`last.pt` is for recovery; `best.pt` is selected by validation objective. Checkpoints contain full optimizer/RNG state and are larger than bare model weights. Restore uses PyTorch `weights_only=True`; unsupported checkpoint formats are rejected. Cross-version RNG/backend equivalence is not guaranteed. The tests compare uninterrupted versus resumed CPU weights exactly.

## Full-data training

First prepare the native dataset and audit the combined manifest as described in [datasets.md](datasets.md). Edit a copied preset to point to that manifest/root and the actual device:

```bash
sanet train --config configs/culane.yaml --output runs/culane-r34-seed17
# Other schedule presets: tusimple.yaml, bdd100k.yaml, foggylane.yaml, once-bev.yaml
```

Presets retain AdamW lr6e-4, cosine, batch32, 320×800, epoch counts 36/70/40/32/20 and loss weights 2/.5/4/1/.5/.5. Weight decay .01, depth/bin/grid sizes, branch counts, embedding margins and augmentation strengths are independent defaults. `device: cuda`, `amp: true` in full presets require a usable CUDA installation. Change batch size and document it if memory is insufficient. Pretrained weights are optional and default false; the manuscript does not disclose initialization. FoggyLane preset is blocked on dataset identification. ONCE requires real calibration and metric annotations; camera-only BEV must be described as camera-only.

## Inference and native evaluators

For unlabelled input, create a canonical manifest with `split: inference`, image paths and capture groups; calibration is required only in BEV mode. Then:

```bash
sanet predict --checkpoint runs/cpu-small/best.pt --manifest data/inference.jsonl --root data --output runs/predictions
sanet export culane --predictions runs/predictions/predictions.jsonl --manifest data/inference.jsonl --output runs/culane-native
# For metric BEV predictions and calibrated records:
sanet export once --predictions runs/bev-predictions/predictions.jsonl --manifest data/once/manifest.jsonl --output runs/once-native
```

CULane text output is in original-image pixels. ONCE output is inverse-transformed into optical camera xyz, and must be placed in the sequence/cam01 layout expected by the pinned official evaluator. TuSimple evaluation writes `tusimple-submission.jsonl` with native requested rows and measured runtime. See the [official TuSimple tool](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py), [SCNN evaluator](https://github.com/XingangPan/SCNN/tree/master/tools/lane_evaluation) and [ONCE evaluator](https://github.com/once-3dlanes/once_3dlanes_benchmark). External evaluators have their own dependencies and version issues; they are not silently patched or claimed to have run on full benchmarks here.

## Hyperparameter search and ablation

```bash
sanet sweep --config configs/cpu-ablation.yaml --output runs/search --learning-rates 0.0003 0.0006 --weight-decays 0.01 --seeds 17
sanet ablate --config configs/cpu-ablation.yaml --output runs/ablation --seeds 17 29 41
```

Search uses validation loss only; it never reads test labels. Image ablations compare backbone+head, +AMPM, +AMPM+ERA. BEV ablations compare dense lift, SFM, SFM+AMPM, SFM+AMPM+ERA. The baseline is this implementation's backbone/head, not original DLNet. The manuscript's extra fifth “complete SA-Net” row has no disclosed change and is omitted. Short 4-epoch comparisons validate the workflow and report variability; they do not establish module superiority or significant benchmark gains.

## Artifacts and interpretation

Each run produces config/environment/data audit, epoch history, status, last/best checkpoints and ledger entries. Each evaluation writes predictions and metrics, including checkpoint/manifest/dataset hashes, measured timing boundaries and the evaluated split. Reports produce actual loss curves, metrics plots and epoch CSV. Manuscript results remain in a separate attributed directory.

Use untouched data and repeated seeds for research claims. The small holdout has only one date group and is too small for generalization or robust confidence intervals. Low positive-class prevalence makes pixel accuracy particularly uninformative; report lane-level results alongside pixel diagnostics. Zero lane F1 in the recorded experiment is a measured failure to recover matched instances, not a successful accuracy reproduction.
