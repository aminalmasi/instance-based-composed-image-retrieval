#!/bin/bash
#SBATCH --job-name=ilias-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

source .venv/bin/activate
instance-cir train-probe --config configs/route_a/train_siglip1_alltrain_infonce.yaml
