#!/bin/bash
#SBATCH --account=brics.b6ar
#SBATCH --job-name=sl_poisson
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --gpus=16
#SBATCH --ntasks-per-node=4
#SBATCH --time=18:00:00
#SBATCH --output=/projects/b6ar/dw22963/GeoChat/experiments/logs/ablations/poisson_eta002_%j.out
#SBATCH --error=/projects/b6ar/dw22963/GeoChat/experiments/logs/ablations/poisson_eta002_%j.err

# ============================================================
# GeoChat Soft Label Ablation: Binomial distribution, eta=0.05
# Isambard-AI, 4 nodes x 4 GH200 = 16 GPUs
# ============================================================

set -eo pipefail

# ---- Modules ----
module load brics/nccl brics/aws-ofi-nccl
module load cuda/12.6

# ---- Conda env ----
source /projects/b6ar/dw22963/miniforge3/etc/profile.d/conda.sh
conda activate geochat

# ---- Distributed env ----
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

# ---- Paths ----
PROJECT_DIR=/projects/b6ar/dw22963/GeoChat
WORKSPACE=/projects/b6ar/dw22963

BASE_MODEL=${WORKSPACE}/models/geochat-7B
VISION_TOWER=${WORKSPACE}/models/clip
DATA_JSON=${WORKSPACE}/data/VRSBench_full/VRSBench_train.json
IMAGE_DIR=${WORKSPACE}/data/VRSBench_full/Images_train
DS_CONFIG=${PROJECT_DIR}/scripts/zero2.json

RUN_TAG=poisson_eta002
OUTPUT_DIR=${WORKSPACE}/models/ablations/lora_${RUN_TAG}

# Ensure a fresh run: train.py auto-resumes when checkpoint-* exists, which
# breaks things if the previous attempt saved a diverged/incompatible state.
rm -rf ${OUTPUT_DIR}

cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export WANDB_API_KEY="..."  # set in your shell or .env, do not commit

echo "============================================"
echo "Poisson eta=0.02  |  4 nodes x 4 GH200"
echo "Date        : $(date)"
echo "Nodes       : $SLURM_JOB_NODELIST"
echo "MASTER_ADDR : $MASTER_ADDR"
echo "Base model  : ${BASE_MODEL}"
echo "Output      : ${OUTPUT_DIR}"
echo "============================================"

srun --mpi=pmi2 bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    echo "[$(hostname)] RANK=$RANK LOCAL_RANK=$LOCAL_RANK WORLD_SIZE=$WORLD_SIZE"
    python '"${PROJECT_DIR}"'/geochat/train/train_multinode.py \
        --deepspeed '"${DS_CONFIG}"' \
        --lora_enable True \
        --model_name_or_path '"${BASE_MODEL}"' \
        --version v1 \
        --data_path '"${DATA_JSON}"' \
        --image_folder '"${IMAGE_DIR}"' \
        --vision_tower '"${VISION_TOWER}"' \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --bf16 True \
        --output_dir '"${OUTPUT_DIR}"' \
        --num_train_epochs 5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 1600 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --dataloader_num_workers 4 \
        --report_to wandb \
        --run_name sl_'"${RUN_TAG}"' \
        --soft_label_enable True \
        --soft_label_distribution poisson \
        --soft_label_eta 0.02 \
        --soft_label_lambda 2.0
'

echo "============================================"
echo "Training finished at $(date)"
echo "Checkpoint: ${OUTPUT_DIR}"
echo "============================================"
