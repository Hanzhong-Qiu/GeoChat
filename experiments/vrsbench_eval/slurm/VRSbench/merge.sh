#!/bin/bash
#SBATCH --account=coms037985
#SBATCH --job-name=merge_lora
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                  # 【关键】申请 64GB CPU 内存
#SBATCH --time=01:00:00            # 合并只要几分钟，申请1小时足够
#SBATCH --output=/user/home/dw22963/work/GeoChat/experiements/logs/soft_merge_%j.out
#SBATCH --error=/user/home/dw22963/work/GeoChat/experiements/logs/soft_merge_%j.err

source ~/initMamba.sh
mamba activate geochat
cd /user/home/dw22963/work/GeoChat/experiements

export PYTHONPATH=/user/home/dw22963/work/GeoChat/code:$PYTHONPATH

python geochat_scripts/merge_lora_weights.py \
    --model-path /user/home/dw22963/work/GeoChat/model/checkpoints/geochat_vrs_5epoch_softlabel_lora \
    --model-base /user/home/dw22963/work/GeoChat/model/geochat-7B \
    --save-model-path /user/home/dw22963/work/GeoChat/model/checkpoints/geochat_vrs_5_softlabel_merged