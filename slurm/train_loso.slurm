#!/bin/bash
#SBATCH --job-name=motion-loso
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/loso-%j.out

set -euo pipefail
mkdir -p logs
source .venv/bin/activate
python scripts/train_loso.py --config configs/default.yaml
