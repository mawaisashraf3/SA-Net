# Completion report

## Delivered software and deployment

The previously empty repository was inspected, then populated through ordinary non-force Git pushes. No existing implementation, branch history or unrelated work was overwritten. The implementation and measured evidence were published on `main` at [commit 1c93c7e1042344af44ec4c1fb930902a9e98cb88](https://github.com/mawaisashraf3/SA-Net/commit/1c93c7e1042344af44ec4c1fb930902a9e98cb88). The GitHub API independently confirmed that remote branch SHA. Subsequent documentation commits do not change the recorded model implementation.

* [Repository and complete README](https://github.com/mawaisashraf3/SA-Net)
* [Source modules](https://github.com/mawaisashraf3/SA-Net/tree/main/src/sanet)
* [Measured engineering checkpoint and model card](../artifacts/MODEL_CARD.md)
* [Tests](https://github.com/mawaisashraf3/SA-Net/tree/main/tests)
* [Research audit](research-audit.md), [mathematical methodology](methodology.md), [architecture and module responsibilities](architecture.md)
* [Dataset engineering](datasets.md), [installation/execution/recovery](reproducibility.md)
* [Measured results and raw evidence](../results/README.md)
* [Original manuscript tables](manuscript/tables.md), [original figures](manuscript/figures.md), [supplied references](manuscript/references.md)
* [GitHub CPU verification workflow](https://github.com/mawaisashraf3/SA-Net/actions)

The repository includes Python packaging/CLI, seven validated YAML configurations, tests, CI, complete research documentation, attributed manuscript assets, actual plots/tables/logs, MIT licensing for new code and separate attribution/rights notices. A full directory listing is [recorded here](project-files.txt); every source file's function is mapped in [architecture.md](architecture.md).

## Manuscript findings and critical evaluation

The complete manuscript was extracted and examined before code development, including all six main sections, subsections, mathematical formulations, experimental tables, seven figures, contribution/data statements and 43 references. The research objective is robust frame-wise lane detection through lane-selected visual/geometric fusion, scene-adaptive context and spatial/channel refinement.

The [audit](research-audit.md) identifies conflicting TuSimple/BDD100K/FoggyLane counts, the ONCE image-resolution discrepancy, missing LiDAR/calibration explanations for image-only benchmarks, figure/text architecture differences, undefined extra ablation configuration, inconsistent DLNet baseline attribution, missing implementation hyperparameters and noncomparable accuracy definitions. Original claims are retained as attributed source material. They are not labeled reproduced. Arithmetic checks cover 25 P/R/F1 rows, with qualifications for macro versus micro aggregation.

## Implemented algorithms and improvements

1. ResNet-34/FPN and an explicitly named compact CPU alternative.
2. SFM with segmentation probabilities, top-K camera-z depth, calibrated pixel-center backprojection, weighted sparse BEV scatter, sparse point/voxel encoding, active modality union and trainable submanifold convolution.
3. AMPM with active-cell pooling, constrained learnable GAP/GMP blend, latent scene routing, dynamic depthwise-kernel mixtures and residual FFN.
4. ERA with parallel directional spatial and channel attention, concatenation and residual projection.
5. Confidence, discriminative instance embeddings, lateral offset and height heads; embedding grouping and image/metric lane reconstruction.
6. Stable/masked multitask training: foreground balancing, positive-normalized regression, explicit embedding margins, collision ignores, sparse observed depth and no fabricated missing labels.
7. Separate image-only and calibrated BEV tasks, per-view annotations, missing-LiDAR support, geometry-aware resize/crop/flip, dataset/asset fingerprints and sequence/duplicate checks.
8. AdamW/cosine, finite/gradient checks, validation-only selection, exact tested CPU recovery, atomic checkpoints, local SQLite ledger, search and repeated-seed ablation workflows.
9. Pixel diagnostics, threshold/cardinality lane matching, TuSimple protocol, diagnostic 3D Chamfer, group bootstrap, native CULane/ONCE exports, synchronized timing and reports.

The portable voxel MLP, factorized dynamic kernels, dense AMPM/ERA, loss normalization and engineering defaults are explicitly distinguished from the manuscript. Sparse lifting/fusion remains sparse until densification; no CUDA sparse-speed claim is inferred. Camera-only datasets receive no invented point clouds or metric calibration.

## Datasets and storage

CULane, TuSimple, legacy/Scalabel BDD100K and ONCE annotation adapters are implemented and exercised with analytic format fixtures. Authoritative sources, formats, licenses/access requirements and unresolved release details are documented. FoggyLane has a preset/canonical schema pending reliable source identification; a dedicated verified native loader is not claimed.

The executed experiment downloaded 40 real TuSimple observations and existing labels from a pinned public mirror. Train/validation/holdout are 24/8/8, partitioned by capture date. The mirror is partial, so acquisition records its actual availability and original annotation indices, URLs and SHA-256 values. Downloaded images are excluded from Git. The holdout is drawn from native training data and is not the official TuSimple test set.

Immutable image/NPY/JSONL inputs remain on disk. A SQLite experiment ledger tracks run status/configs and epoch metrics using primary/foreign keys and parameterized SQL. It is exercised by integration tests. No unnecessary image database or online service is introduced.

## Experiments and verification actually executed

| Execution | Evidence / measured outcome |
|---|---|
| 12-epoch compact image model, seed17, 24/8/8 real observations | [Raw history](../results/cpu-small/history.jsonl); minimum validation objective 3.609037 at epoch11 |
| Eight-image holdout inference/evaluation | Pixel F1 0.346456; lane-IoU F1 **0** at .5/.75; point accuracy 0.597098 under TuSimple protocol. [Complete metrics](../results/cpu-small/metrics.json) |
| Same schedule/seed replay | [Selected model tensors identical](../results/verification/experiment-replay.json); not a separate independent accuracy experiment |
| Three image variants × three seeds × four epochs | [Nine real training/validation runs](../results/ablation/report.md), no test scoring. Mean best validation objective: baseline4.072444, +AMPM3.775999, +AMPM+ERA4.118690 |
| Two learning-rate search trials, four epochs each | [Validation-only selection](../results/search/search.json): 6e-4 selected within this small search |
| 42 tests | [All passed](../results/verification/pytest.txt), [JUnit evidence](../results/verification/junit.xml). Includes both model modes, real mathematical references, gradients, data guards, training/recovery/inference and database integration |
| TuSimple metric parity | [108 analytic cases](../results/verification/tusimple-protocol-parity.json), max absolute difference1.11e-16 against pinned official evaluator |
| Unlabelled real-image inference and native export | [Prediction execution](../results/verification/unlabelled-inference.json), [export record](../results/verification/native-export.txt) |
| Lint, formatting, import/compile and config/link checks | [Lint](../results/verification/lint.txt), [documentation](../results/verification/documentation.txt) |
| Wheel build, isolated installation/import | [Build log](../results/verification/package-build.txt), [install/import log](../results/verification/wheel-install.txt) |

The source commit and source-content fingerprint are recorded per run. Dependencies, CPU environment, random seeds, loss settings, data hashes, best epoch and checkpoint hashes are preserved. Analytic test fixtures are not counted as research datasets or performance observations. CPU and diagnostic metrics do not establish real multimodal accuracy or original benchmark reproduction.

## Figures, tables and reporting

New figures are actual executed loss/schedule curves, holdout metric charts, all-eight-frame lane geometry, and three-seed ablation mean/SD. New tables/JSON/CSV use measured logs. All seven original manuscript figures/tables remain in a separate attributed directory, alongside raw CSV transcriptions and source checksum. The distinction is maintained in the README, reports and assets.

The lane-geometry figure exposes substantial instance-association/continuity failures. The small experiment's zero lane F1 is reported directly. Short ablations are optimization diagnostics and do not justify module superiority or statistical performance claims. The single holdout date group prevents a valid independent-group confidence interval.

## Limitations and work required for scientific validation

Full-data official CULane/TuSimple/BDD100K/ONCE experiments were not run. Original author code, weights, exact split lists and several hyperparameters were unavailable. FoggyLane identity/access/splits remain unresolved. No GPU was available, so real RTX3090 throughput, FLOPs, CUDA/AMP execution and real synchronized camera/LiDAR accuracy are not verified. BEV forward/backward/training/inference were exercised through analytic sensor fixtures, not a real 3D benchmark.

Full original schedules, real calibration and synchronized sensors, official evaluators, matched baseline implementations, untouched independent sequences, repeated-seed statistical comparisons and stronger instance decoding/training remain necessary. The BEV y(x),z(x) parameterization has branching/nonmonotonic-lane limitations; temporal modeling, near-duplicate detection and calibration/sensor degradation studies remain future research. This release supplies a functional, tested research framework and explicit evidence, not a validated autonomous-driving model or a successful accuracy reproduction.
