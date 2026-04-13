import re
import json
import numpy as np
import os

# 确保 eval_utils.py 和当前脚本在同一个目录下
from eval_utils import *

print("开始计算 Visual Grounding 评测指标...")

# 【核心修改 1】：直接指向你生成的预测结果文件
data_path = '/user/home/dw22963/work/GeoChat/experiements/VRSBench_eval/predictions/vg_soft_grounding_predictions.jsonl'

thres_list = [0.5, 0.7]
count = np.zeros(len(thres_list))

cumI = 0
cumU = 0
mean_IoU = 0

total_count = 0
valid_count = 0

use_size_group = False
size_list = ['small', 'medium', 'large']
# 【核心修改 2】：兼容布尔值 True/False (你的数据里 unique 是 true)
#unique_check = [1, 0, True, False] 
#unique_check = [1, True]  # 只保留 Unique 的题目
unique_check = [0, False]  # 只保留 Non-unique 的题目

with open(data_path, 'r') as file:
    for line in file:
        item = json.loads(line.strip())
        img_id = item.get('image_id', 'unknown')
        
        # 【核心修改 3】：修复原作者键名不一致的 bug
        is_unique = item.get('unique', item.get('is_unique', True))
        
        if use_size_group:
            obj_size = item.get('size_group', 'medium')
            obj_size = 'medium' if obj_size == '' else obj_size
        else:
            obj_size = 'medium'

        if not (is_unique in unique_check and (obj_size in size_list)):
            continue
        
        total_count += 1

        # 提取真实的坐标
        integers = re.findall(r'\d+', item.get('ground_truth', ''))
        if len(integers) < 4:
            continue
        gt_bbox = np.array([int(num) for num in integers])[np.newaxis,:]
        
        # 提取模型预测的坐标
        integers = re.findall(r'\d+', item.get('answer', ''))
        if len(integers) < 4:
            pred_bbox = np.array([0, 0, 0, 0])[np.newaxis,:] # 防止模型输出乱码导致报错
        else:
            pred_bbox = np.array([int(num) for num in integers])[np.newaxis,:]
        
        try:
            # 走常规的水平框 HBB 评测逻辑
            gt_bbox_list = gt_bbox[0,:4].tolist()
            pred_bbox_list = pred_bbox[0,:4].tolist()
            
            # 调用 eval_utils.py 里的 computeIoU 函数
            iou_score, I, U = computeIoU(gt_bbox_list, pred_bbox_list, return_iou=True)
            
            mean_IoU += iou_score
            cumI += I
            cumU += U
        
            for ii, thres in enumerate(thres_list):
                if iou_score >= thres:
                    count[ii] += 1
            valid_count += 1

        except Exception as e:
            print(f"Error evaluating {img_id}: {e}")
            print('invalid output', img_id, pred_bbox, gt_bbox, flush=True)

print('-----------------------------------------')
print('number of total/valid samples:', total_count, valid_count)
for ii, thres in enumerate(thres_list):
    print(f'Acc at iou_{thres}: {(count[ii] / total_count * 100):.2f}%', flush=True)

if cumU > 0:
    print(f'meanIoU: {(mean_IoU/total_count * 100):.2f}%, cumIoU: {(cumI/cumU * 100):.2f}%', flush=True)
else:
    print("Failed to calculate IoU.")