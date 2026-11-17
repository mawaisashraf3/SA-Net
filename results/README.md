# Executed results and verification evidence

The committed CPU experiment uses real TuSimple images and existing annotations. It is a limited engineering experiment, not reproduction of the supplied manuscript. Original results are kept separately in [docs/manuscript](../docs/manuscript/tables.md).

The compact image-only model trained for 12 epochs on 24 images, selected a checkpoint using 8 validation images, and evaluated 8 holdout images from a third capture date. Observed pixel F1 was 0.3465; lane-instance F1 was 0 at both diagnostic IoU thresholds. This records an instance-detection failure and does not validate the architecture's accuracy claims. Complete metrics, logs, provenance and verification evidence are recorded below as the final verification proceeds.

* [Measured report](cpu-small/report.md), [metrics](cpu-small/metrics.json), [predictions](cpu-small/predictions.jsonl)
* [Raw epoch history](cpu-small/history.jsonl), [configuration](cpu-small/config.json), [environment](cpu-small/environment.json)
* [Data audit](cpu-small/data-audit.json), [pinned acquisition provenance](cpu-small/data-provenance.json)
* [Test output](verification/pytest.txt)

![Actual holdout metrics](cpu-small/holdout-metrics.png)
