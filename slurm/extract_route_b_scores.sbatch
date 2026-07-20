#!/bin/bash
#SBATCH --job-name=icir-rzen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

: "${MODEL_CACHE_ROOT:?Set MODEL_CACHE_ROOT}"

source .venv/bin/activate
python scripts/extract_rzen_reranker_scores.py \
  --rankings outputs/first_stage/topk_rankings.csv \
  --descriptors artifacts/icir_exact_line2_instance_descriptors.csv \
  --query-image-features features/icir/rzen/query_image_features.npy \
  --query-text-features features/icir/rzen/query_text_features.npy \
  --query-metadata features/icir/rzen/query_metadata.csv \
  --model-root "$MODEL_CACHE_ROOT/RzenEmbed" \
  --output-dir outputs/route_b_scores
