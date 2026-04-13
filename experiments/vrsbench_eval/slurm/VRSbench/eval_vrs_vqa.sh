#!/bin/bash
#SBATCH --account=coms037985
#SBATCH --job-name=eval_vrs5_vqa
#SBATCH --nodes=1
#SBATCH --partition=mlcnu
#SBATCH --gres=gpu:a100-sxm4-40gb:4
#SBATCH --time=24:00:00
#SBATCH --output=/user/work/dw22963/GeoChat/Training/logs/eval_vrs5_vqa_%j.out
#SBATCH --error=/user/work/dw22963/GeoChat/Training/logs/eval_vrs5_vqa_%j.err

export BASHRCSOURCED=1
set +u
set -eo pipefail

mkdir -p /user/work/$USER/GeoChat/logs

source ~/initMamba.sh
mamba activate geochat
module load cuda/11.8.0-rgxs
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export HF_HOME=/user/work/dw22963/hf_cache/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/

cd /user/work/$USER/GeoChat

# 【核心修改】：用循环同时启动4个进程
# --num-chunks 4 表示将数据集等分为 4 份
# --chunk-idx $IDX 表示当前进程处理第几份数据
# 最后的 & 符号表示让它们在后台同时运行
for IDX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$IDX python geochat/eval/batch_geochat_vqa.py \
      --model-path /user/work/dw22963/GeoChat/Training/checkpoints/geochat_vrs_5_merged \
      --question-file /user/work/dw22963/GeoChat/Training/VRSBench/VRSBench_EVAL_vqa.json \
      --answers-file /user/work/dw22963/GeoChat/Training/VRSBench/vrs_vqa_predictions_${IDX}.jsonl \
      --image-folder /user/work/dw22963/GeoChat/Training/VRSBench/Images_val \
      --num-chunks 4 \
      --chunk-idx $IDX &
done
 
# 【核心修改】：等待所有的 4 个后台进程运行完毕
wait

# 【核心修改】：合并 4 张卡生成的结果文件
cat /user/work/dw22963/GeoChat/Training/VRSBench/vrs_vqa_predictions_*.jsonl > /user/work/dw22963/GeoChat/Training/VRSBench/vrs_vqa_predictions.jsonl

echo "全部 4 张卡推理完毕，并已合并为最终文件！"