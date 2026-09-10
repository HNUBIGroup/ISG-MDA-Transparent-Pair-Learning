#!/usr/bin/env bash
#SBATCH --job-name=isg-mda-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

project_root="${PROJECT_ROOT:-$PWD}"
data_root="${DATA_ROOT:?Set DATA_ROOT to the prepared data directory}"
output_root="${OUTPUT_ROOT:-$project_root/outputs}"

cd "$project_root"
export DATA_ROOT="$data_root"
python scripts/training/train.py \
  --config configs/default.yaml \
  --train-pairs "$data_root/splits/train.tsv" \
  --drug-table "$data_root/drugs.tsv" \
  --mirna-table "$data_root/mirnas.tsv" \
  --output-dir "$output_root/training"
