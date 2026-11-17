"""CLI for acquisition, conversion, audit, training, prediction, search and reports."""

import argparse
import json
import logging
from pathlib import Path

from sanet.config import load_config
from sanet.io import write_json, write_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sanet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("acquire-small")
    p.add_argument("--root", default="data/tusimple-small")
    for name, count in [("train", 24), ("val", 8), ("test", 8)]:
        p.add_argument(f"--{name}-count", type=int, default=count)
    p = sub.add_parser("convert")
    p.add_argument("format", choices=["tusimple", "culane", "bdd100k", "once"])
    p.add_argument("--labels", required=True, help="JSON(L) or split listing")
    p.add_argument("--root", default="data")
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--image-prefix", default="")
    p.add_argument("--transform", help="JSON optical camera to ego 4x4 matrix")
    p.add_argument("--output", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--mode", choices=["image", "bev"], default="image")
    p.add_argument("--output")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    for name in ["evaluate", "predict"]:
        p = sub.add_parser(name)
        for field in ["checkpoint", "manifest", "root", "output"]:
            p.add_argument(f"--{field}", required=True)
        p.add_argument("--split", default="test" if name == "evaluate" else "inference")
        p.add_argument("--device", default="cpu")
    p = sub.add_parser("export")
    p.add_argument("format", choices=["culane", "once"])
    p.add_argument("--predictions", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--evaluation")
    p.add_argument("--output")
    for name in ["sweep", "ablate"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
        if name == "sweep":
            p.add_argument("--learning-rates", type=float, nargs="+", default=[0.0003, 0.0006])
            p.add_argument("--weight-decays", type=float, nargs="+", default=[0.01])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "acquire-small":
        from sanet.data.acquire import acquire_small

        result = acquire_small(args.root, args.train_count, args.val_count, args.test_count)
    elif args.command == "convert":
        from sanet.data import adapters

        if args.format == "tusimple":
            records = adapters.tusimple(args.labels, args.split)
        elif args.format == "culane":
            records = adapters.culane(args.root, args.labels, args.split)
        elif args.format == "bdd100k":
            records = adapters.bdd100k(args.labels, args.split, args.image_prefix)
        else:
            if not args.transform:
                parser.error("ONCE needs --transform; extrinsics are never inferred")
            records = adapters.once(
                args.root,
                args.labels,
                args.split,
                args.image_prefix,
                json.loads(Path(args.transform).read_text()),
            )
        write_jsonl(args.output, records)
        result = {"records": len(records), "output": args.output}
    elif args.command == "audit":
        from sanet.data.audit import audit_manifest

        result = audit_manifest(args.manifest, args.root, args.mode)
        if args.output:
            write_json(args.output, result)
    elif args.command == "train":
        from sanet.engine.train import train

        history = train(load_config(args.config), args.output, args.resume, args.stop_after)
        result = {"recorded_epochs": len(history), "output": args.output}
    elif args.command in {"evaluate", "predict"}:
        from sanet.engine.evaluate import evaluate

        result = evaluate(
            args.checkpoint,
            args.manifest,
            args.root,
            args.split,
            args.output,
            args.device,
            score=args.command == "evaluate",
        )
    elif args.command == "export":
        from sanet.evaluation.export import export_predictions

        result = export_predictions(args.predictions, args.manifest, args.output, args.format)
    elif args.command == "report":
        from sanet.evaluation.report import report_run

        result = report_run(args.run, args.evaluation, args.output)
    elif args.command == "sweep":
        from sanet.engine.sweep import sweep

        result = sweep(
            load_config(args.config),
            args.output,
            args.learning_rates,
            args.weight_decays,
            args.seeds,
        )
    else:
        from sanet.engine.sweep import ablate

        result = ablate(load_config(args.config), args.output, args.seeds)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
