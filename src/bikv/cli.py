from __future__ import annotations

import argparse
import json
from bikv.config import ProjectConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Boundary-Interaction KV research runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "qualify", "label", "recapture", "train", "evaluate", "report", "bundle"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, help="Path to a JSON experiment config")
    return parser


def main() -> None:
    from bikv.runner import evaluate, label_interactions, make_store, prepare_dataset, recapture_features, train_predictor

    args = build_parser().parse_args()
    config = ProjectConfig.load(args.config)
    if args.command == "prepare":
        result = {"data_path": str(prepare_dataset(config))}
    elif args.command == "qualify":
        from bikv.runner import qualify_backbone

        result = qualify_backbone(config)
    elif args.command == "label":
        result = label_interactions(config)
    elif args.command == "recapture":
        result = recapture_features(config)
    elif args.command == "train":
        result = train_predictor(config)
    elif args.command == "evaluate":
        result = evaluate(config)
    elif args.command == "report":
        from bikv.analysis import build_report

        result = build_report(config)
    else:
        result = {"bundle": str(make_store(config).bundle())}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
