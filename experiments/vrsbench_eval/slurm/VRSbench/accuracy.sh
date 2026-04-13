#!/bin/bash
#SBATCH --account=coms037985
#SBATCH --job-name=vrs_gpt_eval_5
#SBATCH --partition=short       # 使用 short 分区，资源很多，排队极快
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16      # 申请 16 个 CPU 核心配合多线程
#SBATCH --time=03:00:00         # 多线程下几十分钟就跑完了，申请 3 小时足矣
#SBATCH --output=/user/work/dw22963/GeoChat/Training/logs/eval_gpt_5_%j.out
#SBATCH --error=/user/work/dw22963/GeoChat/Training/logs/eval_gpt_5_%j.err

source ~/initMamba.sh
mamba activate geochat

cd /user/work/dw22963/GeoChat/Training/VRSgit/VRSBench/eval_fianl
python eval_vqa_gpt.py