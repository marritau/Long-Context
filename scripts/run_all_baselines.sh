#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/exp02_retrieval_vs_full.yaml}"
echo "Baseline runner scaffold. Config: ${CONFIG}"
echo "Wire dataset paths and generation backend before full baseline execution."
