# Compact engineering checkpoint

`cpu-small-best.pt` is a full training checkpoint from the executed, independently implemented SA-Net image-mode experiment. It is provided for pipeline inspection and replay. It is **not** an original author SA-Net weight file or a validated driving model.

* Architecture: compact camera backbone/FPN, width24, embedding dimension4, AMPM+ERA, image-plane instance head.
* Data: 24 real TuSimple training images, 8 validation images; capture dates partitioned as recorded in [provenance](../results/cpu-small/data-provenance.json).
* Training: 12 epochs, seed17, AdamW6e-4, cosine, 96×240, CPU. Checkpoint selected by minimum validation objective3.609037 at epoch11 (stored zero-based10).
* Holdout: 8 real images from a third date, drawn from the native training release; not the official benchmark test set. Pixel F1 .346456; diagnostic lane-instance F1 **0** at IoU .5/.75. Instance association/continuity is poor.
* Tested checkpoint runtime: torch2.15.0+cpu and torchvision0.29.2+cpu; [tested direct versions](../constraints-tested.txt).
* Contents: model, optimizer/scheduler/scaler state, RNG/sampler state, run config/history, manifest and asset fingerprints. Restore uses `weights_only=True`.
* Integrity: [SHA256SUMS](SHA256SUMS); the same checkpoint hash appears in [evaluation metrics](../results/cpu-small/metrics.json).

After installation and real data acquisition:

```bash
sanet evaluate --checkpoint artifacts/cpu-small-best.pt --manifest data/tusimple-small/manifest.jsonl --root data/tusimple-small --split test --output runs/released-evaluation
```

For unlabelled inference, use a manifest with `split: inference` and `sanet predict`; target labels are not model inputs. Do not interpret the checkpoint as a successful manuscript reproduction. See [complete limitations](../docs/completion-report.md) and [attribution/rights](../NOTICE.md).
