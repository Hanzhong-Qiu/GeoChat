#!/bin/bash
#SBATCH --account=brics.b6ar
#SBATCH --job-name=eval_sl_vg
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4                        # ← 改：2 → 4
#SBATCH --time=04:00:00
#SBATCH --output=/projects/b6ar/dw22963/logs/eval_softlabel_vg_%j.out
#SBATCH --error=/projects/b6ar/dw22963/logs/eval_softlabel_vg_%j.err

set -eo pipefail

source /projects/b6ar/dw22963/miniforge3/bin/activate
conda activate geochat

PROJECT_DIR=/projects/b6ar/dw22963/GeoChat
WORKSPACE=/projects/b6ar/dw22963

MERGED_MODEL=${WORKSPACE}/models/geochat_vrs_softlabel_newcode_merged
QUESTION_FILE=${PROJECT_DIR}/experiments/vrsbench_eval/ground_truth/VRSBench_EVAL_referring.json
IMAGE_FOLDER=${WORKSPACE}/data/VRSBench_full/Images_val
PRED_DIR=${PROJECT_DIR}/experiments/vrsbench_eval/predictions
PRED_PREFIX=softlabel_newcode_vg_predictions

cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

echo "============================================"
echo "Visual Grounding Evaluation (Isambard, 4-GPU)"
echo "Date: $(date)"
echo "Model: ${MERGED_MODEL}"
echo "============================================"

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

echo "============================================"
echo "Done at $(date), total predictions: $(wc -l < ${PRED_DIR}/${PRED_PREFIX}.jsonl)"
echo "============================================"
