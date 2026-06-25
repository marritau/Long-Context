#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
python -m pearl.experiments.run_ablation --config "${CONFIG}"
