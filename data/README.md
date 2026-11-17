# External data

See [dataset engineering](../docs/datasets.md) for sources, formats, terms and conversion commands. Data, weights and experiment databases are ignored by Git.

```bash
sanet acquire-small --root data/tusimple-small
sanet audit --manifest data/tusimple-small/manifest.jsonl --root data/tusimple-small
```

This downloads real existing observations for a small engineering experiment. It does not create a synthetic dataset or reproduce the official benchmark split. Full-data experiments need separately acquired releases and validated calibration for BEV mode.
