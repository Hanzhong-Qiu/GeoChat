#!/bin/bash
#SBATCH --account=coms037985
#SBATCH --job-name=vrs_geochat
#SBATCH --partition=mlcnu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --time=02-24:00:00
#SBATCH --output=/user/work/dw22963/GeoChat/Training/logs/train_%j.out
#SBATCH --error=/user/work/dw22963/GeoChat/Training/logs/train_%j.err

export BASHRCSOURCED=1
set +u
set -eo pipefail

mkdir -p /user/work/dw22963/GeoChat/Training/logs
mkdir -p /user/work/dw22963/GeoChat/Training/checkpoints

source ~/initMamba.sh
mamba activate geochat
module load cuda/11.8.0-rgxs
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export WANDB_API_KEY="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"
#export HF_HOME=/user/work/dw22963/hf_cache/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/

cd /user/work/$USER/GeoChat/
PROMPT_VERSION=v1

 deepspeed --master_port=$((RANDOM + 10000)) --num_gpus=4 geochat/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /user/work/dw22963/GeoChat/geochat-7B \
    --version $PROMPT_VERSION \
    --data_path /user/work/dw22963/GeoChat/Training/VRSBench/VRSBench_train.json \
    --image_folder /user/work/dw22963/GeoChat/Training/VRSBench/Images_train  \
    --vision_tower /user/work/dw22963/hf_cache/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir /user/work/dw22963/GeoChat/Training/checkpoints/geochat_vrs_1epoch \
    --num_train_epochs 1 \
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
    --report_to wandb
