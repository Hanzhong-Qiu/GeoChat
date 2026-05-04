#!/bin/bash
#SBATCH --account=brics.b6ar
#SBATCH --job-name=merge_hard_baseline
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/projects/b6ar/dw22963/GeoChat/experiments/logs/ablations/merge_hard_baseline_eta005_%j.out
#SBATCH --error=/projects/b6ar/dw22963/GeoChat/experiments/logs/ablations/merge_hard_baseline_eta005_%j.err

set -eo pipefail

# ---- Conda env ----
source /projects/b6ar/dw22963/miniforge3/etc/profile.d/conda.sh
conda activate geochat

# ---- Paths ----
PROJECT_DIR=/projects/b6ar/dw22963/GeoChat
WORKSPACE=/projects/b6ar/dw22963

RUN_TAG=hard_baseline_isambard
LORA_CKPT=${WORKSPACE}/models/ablations/lora_${RUN_TAG}
BASE_MODEL=${WORKSPACE}/models/geochat-7B
MERGED_OUT=${WORKSPACE}/models/ablations/geochat_merged_${RUN_TAG}

# Remove prior merge output to ensure clean overwrite
rm -rf ${MERGED_OUT}

cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================"
echo "Merge LoRA (${RUN_TAG})"
echo "Date: $(date)"
echo "LoRA  : ${LORA_CKPT}"
echo "Base  : ${BASE_MODEL}"
echo "Output: ${MERGED_OUT}"
echo "============================================"

python scripts/merge_lora_weights.py \
    --model-path ${LORA_CKPT} \
    --model-base ${BASE_MODEL} \
    --save-model-path ${MERGED_OUT}

echo "============================================"
echo "Merge finished at $(date)"
ls -lh ${MERGED_OUT}/
echo "============================================"
