#!/bin/bash
#SBATCH --account=coms037985
#SBATCH --job-name=eval_soft_vg_grounding
#SBATCH --nodes=1
#SBATCH --partition=mlcnu
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --time=12:00:00
#SBATCH --output=/user/home/dw22963/work/GeoChat/experiements/logs/eval_soft_referring_%j.out
#SBATCH --error=/user/home/dw22963/work/GeoChat/experiements/logs/eval_soft_referring_%j.err

set +u
set -eo pipefail

source ~/initMamba.sh
mamba activate geochat
module load cuda/11.8.0-rgxs
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1

cd /user/work/dw22963/GeoChat/code

export PYTHONPATH="/user/work/dw22963/GeoChat/code:$PYTHONPATH"

# 【注意】请确认你的问题文件（question-file）的具体名称，这里假设为 VRSBench_EVAL_referring.json
# 【注意】请确认使用的 python 脚本名称，这里假设为 batch_geochat_referring.py
for IDX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$IDX python geochat/eval/batch_geochat_grounding.py \
      --model-path /user/home/dw22963/work/GeoChat/model/checkpoints/geochat_vrs_5_softlabel_merged \
      --question-file /user/home/dw22963/work/GeoChat/experiements/VRSBench_eval/ground_truth/VRSBench_EVAL_referring.json \
      --answers-file /user/home/dw22963/work/GeoChat/experiements/VRSBench_eval/predictions/vg_soft_grounding_predictions_${IDX}.jsonl \
      --image-folder /user/work/dw22963/GeoChat/images/VRSimages/Images_val \
      --num-chunks 4 \
      --batch_size 4 \
      --chunk-idx $IDX &
done

wait

# 合并 4 张卡的结果
cat /user/home/dw22963/work/GeoChat/experiements/VRSBench_eval/predictions/vg_soft_grounding_predictions_*.jsonl > /user/home/dw22963/work/GeoChat/experiements/VRSBench_eval/predictions/vg_soft_grounding_predictions.jsonl

echo "Grounding 推理完毕！"