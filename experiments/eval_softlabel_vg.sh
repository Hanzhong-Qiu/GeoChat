#!/bin/bash
#SBATCH --account=coms037985
#SBATCH --job-name=eval_sl_vg
#SBATCH --partition=mlcnu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --time=12:00:00
#SBATCH --output=/user/work/dw22963/Newgeochat/GeoChat/experiments/logs/eval_softlabel_vg_%j.out
#SBATCH --error=/user/work/dw22963/Newgeochat/GeoChat/experiments/logs/eval_softlabel_vg_%j.err

# ============================================================
# Visual Grounding Evaluation on VRSBench
# ============================================================
# Evaluates the merged soft-label GeoChat model on VRSBench
# referring expression task using 4 GPUs in parallel.
#
# Usage:
#   sbatch experiments/eval_softlabel_vg.sh
# ============================================================

export BASHRCSOURCED=1
set +u
set -eo pipefail

source ~/initMamba.sh
mamba activate geochat
module load cuda/11.8.0-rgxs
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1

# ---- Paths ----
PROJECT_DIR=/user/work/dw22963/Newgeochat/GeoChat
OLD_GEOCHAT=/user/work/dw22963/GeoChat

MERGED_MODEL=${OLD_GEOCHAT}/model/checkpoints/geochat_vrs_softlabel_newcode_merged
QUESTION_FILE=${OLD_GEOCHAT}/experiements/VRSBench_eval/ground_truth/VRSBench_EVAL_referring.json
IMAGE_FOLDER=${OLD_GEOCHAT}/images/VRSimages/Images_val
PRED_DIR=${OLD_GEOCHAT}/experiements/VRSBench_eval/predictions
PRED_PREFIX=softlabel_newcode_vg_predictions

cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

echo "============================================"
echo "Visual Grounding Evaluation"
echo "Date: $(date)"
echo "Model: ${MERGED_MODEL}"
echo "Question file: ${QUESTION_FILE}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Predictions: ${PRED_DIR}/${PRED_PREFIX}.jsonl"
echo "============================================"

# Run 2-GPU parallel evaluation (one chunk per GPU)
for IDX in 0 1; do
    CUDA_VISIBLE_DEVICES=$IDX python geochat/eval/batch_geochat_grounding.py \
      --model-path ${MERGED_MODEL} \
      --question-file ${QUESTION_FILE} \
      --answers-file ${PRED_DIR}/${PRED_PREFIX}_${IDX}.jsonl \
      --image-folder ${IMAGE_FOLDER} \
      --num-chunks 2 \
      --batch_size 4 \
      --chunk-idx $IDX &
done

# Wait for all 2 processes to finish
wait

echo "All 2 GPU processes completed. Merging prediction files..."

# Merge 2 chunk files into one
cat ${PRED_DIR}/${PRED_PREFIX}_*.jsonl > ${PRED_DIR}/${PRED_PREFIX}.jsonl

echo "============================================"
echo "Evaluation completed at $(date)"
echo "Predictions saved to: ${PRED_DIR}/${PRED_PREFIX}.jsonl"
echo "Total predictions: $(wc -l < ${PRED_DIR}/${PRED_PREFIX}.jsonl)"
echo "============================================"
