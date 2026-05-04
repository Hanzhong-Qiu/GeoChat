#!/bin/bash
#SBATCH --account=brics.b6ar
#SBATCH --job-name=eval_binomial
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/b6ar/dw22963/GeoChat/experiments/logs/ablations/eval_binomial_eta005_%j.out
#SBATCH --error=/projects/b6ar/dw22963/GeoChat/experiments/logs/ablations/eval_binomial_eta005_%j.err

set -eo pipefail

source /projects/b6ar/dw22963/miniforge3/etc/profile.d/conda.sh
conda activate geochat

PROJECT_DIR=/projects/b6ar/dw22963/GeoChat
WORKSPACE=/projects/b6ar/dw22963

RUN_TAG=binomial_eta005
MERGED_MODEL=${WORKSPACE}/models/ablations/geochat_merged_${RUN_TAG}
QUESTION_FILE=${PROJECT_DIR}/experiments/vrsbench_eval/ground_truth/VRSBench_EVAL_referring.json
IMAGE_FOLDER=${WORKSPACE}/data/VRSBench_full/Images_val
PRED_DIR=${PROJECT_DIR}/experiments/vrsbench_eval/predictions
PRED_PREFIX=${RUN_TAG}_vg_predictions

cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

echo "============================================"
echo "VG Eval (${RUN_TAG}, 4-GPU)"
echo "Model: ${MERGED_MODEL}"
echo "Date : $(date)"
echo "============================================"

# 4-GPU parallel inference, one chunk per GPU
for IDX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$IDX python geochat/eval/batch_geochat_grounding.py \
        --model-path ${MERGED_MODEL} \
        --question-file ${QUESTION_FILE} \
        --answers-file ${PRED_DIR}/${PRED_PREFIX}_${IDX}.jsonl \
        --image-folder ${IMAGE_FOLDER} \
        --num-chunks 4 \
        --batch_size 4 \
        --chunk-idx $IDX &
done

wait

cat ${PRED_DIR}/${PRED_PREFIX}_*.jsonl > ${PRED_DIR}/${PRED_PREFIX}.jsonl
rm ${PRED_DIR}/${PRED_PREFIX}_[0-3].jsonl

echo "============================================"
echo "Done at $(date)"
echo "Predictions: $(wc -l < ${PRED_DIR}/${PRED_PREFIX}.jsonl) lines"
echo "File: ${PRED_DIR}/${PRED_PREFIX}.jsonl"
echo "============================================"
