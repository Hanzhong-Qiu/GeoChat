#!/bin/bash
#SBATCH --account=coms037985
#SBATCH --job-name=merge_sl_lora
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/user/work/dw22963/Newgeochat/GeoChat/experiments/logs/merge_softlabel_%j.out
#SBATCH --error=/user/work/dw22963/Newgeochat/GeoChat/experiments/logs/merge_softlabel_%j.err

# ============================================================
# Merge LoRA adapter weights with GeoChat-7B base model
# ============================================================
# Input:  LoRA checkpoint (adapter_model.bin + non_lora_trainables.bin)
# Output: Full merged model ready for inference
#
# Usage:
#   sbatch experiments/merge_softlabel_lora.sh
# ============================================================

export BASHRCSOURCED=1
set +u
set -eo pipefail

source ~/initMamba.sh
mamba activate geochat
export HF_HUB_OFFLINE=1

# ---- Paths ----
PROJECT_DIR=/user/work/dw22963/Newgeochat/GeoChat
LORA_CHECKPOINT=/user/work/dw22963/GeoChat/model/checkpoints/geochat_vrs_softlabel_newcode_lora
BASE_MODEL=/user/work/dw22963/GeoChat/model/geochat-7B
MERGED_OUTPUT=/user/work/dw22963/GeoChat/model/checkpoints/geochat_vrs_softlabel_newcode_merged

cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

echo "============================================"
echo "Merging LoRA weights"
echo "Date: $(date)"
echo "LoRA checkpoint: ${LORA_CHECKPOINT}"
echo "Base model: ${BASE_MODEL}"
echo "Output: ${MERGED_OUTPUT}"
echo "============================================"

python scripts/merge_lora_weights.py \
    --model-path ${LORA_CHECKPOINT} \
    --model-base ${BASE_MODEL} \
    --save-model-path ${MERGED_OUTPUT}

echo "============================================"
echo "Merge completed at $(date)"
echo "Merged model saved to: ${MERGED_OUTPUT}"
echo "Contents:"
ls -lh ${MERGED_OUTPUT}/
echo "============================================"
