#!/bin/bash
#SBATCH --job-name=motion-prep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare-%j.out

set -euo pipefail
mkdir -p logs
source .venv/bin/activate
python scripts/prepare_data.py --config configs/default.yaml
