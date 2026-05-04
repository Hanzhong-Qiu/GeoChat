#!/bin/bash
# Run compute_metrics on every available predictions file and aggregate into one table.
#
# Output columns: tag | split | N | Acc@0.3 | Acc@0.5 | Acc@0.7 | Acc@0.9 | meanIoU | cumIoU

set -eo pipefail

source /projects/b6ar/dw22963/miniforge3/etc/profile.d/conda.sh
conda activate geochat

PROJECT_DIR=/projects/b6ar/dw22963/GeoChat
EVAL_DIR=${PROJECT_DIR}/experiments/vrsbench_eval
PRED_DIR=${EVAL_DIR}/predictions
CODE_DIR=${EVAL_DIR}/eval_code
OUTPUT=${EVAL_DIR}/RESULTS_ALL_MODELS.tsv

cd ${CODE_DIR}

# Models to evaluate: (tag, prediction_file_basename)
# Order: hard baseline, main soft result, then ablations alphabetical
declare -a MODELS=(
    "hard_baseline_bluepebble:vrs_grounding_predictions"
    "triangular_eta008_bluepebble:softlabel_newcode_vg_predictions"
    "hard_baseline_isambard:hard_baseline_isambard_vg_predictions"
    "triangular_eta008_isambard:triangular_eta008_isambard_vg_predictions"
    "binomial_eta002:binomial_eta002_vg_predictions"
    "binomial_eta005:binomial_eta005_vg_predictions"
    "binomial_eta010:binomial_eta010_vg_predictions"
    "poisson_eta002:poisson_eta002_vg_predictions"
    "triangular_eta005:triangular_eta005_vg_predictions"
    "triangular_eta010:triangular_eta010_vg_predictions"
    "uniform_eta005:uniform_eta005_vg_predictions"
)

# Write header once
python compute_metrics.py ${PRED_DIR}/softlabel_newcode_vg_predictions.jsonl \
    --tag triangular_eta008 --header | head -1 > ${OUTPUT}

for entry in "${MODELS[@]}"; do
    tag="${entry%%:*}"
    base="${entry##*:}"
    file="${PRED_DIR}/${base}.jsonl"
    if [ ! -f "$file" ]; then
        echo "[skip] $tag: $file not found"
        continue
    fi
    echo "[run]  $tag: $file"
    python compute_metrics.py "$file" --tag "$tag" | tail -n +1 | grep -v "^tag" >> ${OUTPUT}
done

echo ""
echo "=== Aggregated results: ${OUTPUT} ==="
column -t -s $'\t' ${OUTPUT}
