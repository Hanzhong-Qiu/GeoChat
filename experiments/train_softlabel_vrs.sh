#!/bin/bash
#SBATCH --account=coms037985
#SBATCH --job-name=geochat_sl
#SBATCH --partition=mlcnu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --time=02-24:00:00
#SBATCH --output=/user/work/dw22963/Newgeochat/GeoChat/experiments/logs/train_softlabel_%j.out
#SBATCH --error=/user/work/dw22963/Newgeochat/GeoChat/experiments/logs/train_softlabel_%j.err

# ============================================================
# GeoChat + Soft Labeling Training Script
# ============================================================
# Description:
#   Fine-tune GeoChat-7B on VRSBench with soft labeling enabled.
#   Uses the new soft_label_loss module from Newgeochat codebase.
#
# Soft Labeling Paper:
#   "Enhancing Numerical Prediction of MLLMs with Soft Labeling"
#   (ICCV 2025, Wang et al.)
#
# Usage:
#   sbatch experiments/train_softlabel_vrs.sh
# ============================================================

export BASHRCSOURCED=1
set +u
set -eo pipefail

# ---- Environment setup ----
source ~/initMamba.sh
mamba activate geochat
module load cuda/11.8.0-rgxs
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
# export WANDB_API_KEY="..."  # set in your shell or .env, do not commit

# ---- Paths ----
PROJECT_DIR=/user/work/dw22963/Newgeochat/GeoChat
OLD_GEOCHAT=/user/work/dw22963/GeoChat

# Base model: original GeoChat-7B
BASE_MODEL=${OLD_GEOCHAT}/model/geochat-7B

# VRSBench dataset
DATA_JSON=${OLD_GEOCHAT}/experiements/VRSBench_eval/ground_truth/VRSBench_train.json
IMAGE_DIR=${OLD_GEOCHAT}/images/VRSimages/Images_train

# Vision tower (offline cached CLIP)
VISION_TOWER=/user/work/dw22963/hf_cache/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1

# DeepSpeed config
DS_CONFIG=${PROJECT_DIR}/scripts/zero2.json

# Output checkpoint directory
OUTPUT_DIR=${OLD_GEOCHAT}/model/checkpoints/geochat_vrs_softlabel_newcode_lora

# ---- Training ----
cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

PROMPT_VERSION=v1

echo "============================================"
echo "Starting GeoChat Soft Label Training"
echo "Date: $(date)"
echo "Base model: ${BASE_MODEL}"
echo "Dataset: ${DATA_JSON}"
echo "Output: ${OUTPUT_DIR}"
echo "Soft label: triangular, eta=0.08, lambda=2.0"
echo "============================================"

deepspeed --master_port=$((RANDOM + 10000)) --num_gpus=4 geochat/train/train_mem.py \
    --deepspeed ${DS_CONFIG} \
    --lora_enable True \
    --model_name_or_path ${BASE_MODEL} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_JSON} \
    --image_folder ${IMAGE_DIR} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 9 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1600 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --soft_label_enable True \
    --soft_label_distribution triangular \
    --soft_label_eta 0.08 \
    --soft_label_lambda 2.0

echo "============================================"
echo "Training finished at $(date)"
echo "Checkpoint saved to: ${OUTPUT_DIR}"
echo "============================================"
