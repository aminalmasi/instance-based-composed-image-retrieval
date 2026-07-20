#!/bin/bash
#SBATCH --job-name=icir-siglip2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

: "${ICIR_DATA_ROOT:?Set ICIR_DATA_ROOT}"
: "${MODEL_CACHE_ROOT:?Set MODEL_CACHE_ROOT}"

source .venv/bin/activate
python scripts/extract_siglip_features.py \
  --data-root "$ICIR_DATA_ROOT" \
  --model-id "$MODEL_CACHE_ROOT/siglip2-so400m-patch16-384" \
  --subjects corpora/generic_subjects.csv \
  --output-dir features/icir/siglip2_so400m
