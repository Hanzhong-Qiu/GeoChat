import json
import os
import threading
from tqdm import tqdm
import numpy as np
import inflect
from openai import OpenAI
import concurrent.futures
import time

# 1. 设置你的 OpenAI API Key 
if not os.environ.get('OPENAI_API_KEY'):
    raise RuntimeError("Please set OPENAI_API_KEY environment variable before running.")
client = OpenAI()

def check_match_with_gpt(question, ground_truth, predicted, retries=3):
    prompt = f"Question: {question}\nGround Truth Answer: {ground_truth}\nPredicted Answer: {predicted}\nDoes the predicted answer match the ground truth? Answer 1 for match and 0 for not match. Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.\n"
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2) # 遇到 API 频率限制时稍微等一下再试
            else:
                print(f"API Error after {retries} retries: {e}")
                return "0"

# 2. 设置输入和输出的文件路径
input_file = '/user/work/dw22963/GeoChat/Training/VRSBench/vrs_vqa_predictions.jsonl'
output_file = '/user/work/dw22963/GeoChat/Training/VRSBench/vrs_vqa_gpt4omini_evaluated_5.jsonl'

print("Loading predictions...")
with open(input_file, 'r') as f:
    qa_list = [json.loads(line) for line in f.readlines()]

# 文件写入锁，防止多线程同时写文件导致混乱
file_lock = threading.Lock()
results = []

def process_single_qa(qa, ii):
    question = qa.get('question', '')
    ground_truth = qa.get('ground_truth', '').lower()
    predicted = qa.get('answer', '').lower()
    
    if ground_truth in predicted:
        match_result = '1'
    elif ground_truth in ['yes', 'no'] + list(map(str, range(100))):
        match_result = '1' if ground_truth == predicted else '0'
    else:
        match_result = check_match_with_gpt(question, ground_truth, predicted)
        
    result = {
        'question_id': qa.get('question_id', ii),
        'image_id': qa.get('image_id', ''),
        "type": qa.get('type', ''),
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": match_result,
    }
    return result

print("Evaluating with GPT (Multithreading Enabled)...")
# 开启 20 个并发线程（可以根据你的 API 限流等级调大或调小）
MAX_WORKERS = 30

with open(output_file, 'w') as f:
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_qa, qa, ii): qa for ii, qa in enumerate(qa_list)}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(qa_list)):
            res = future.result()
            results.append(res)
            # 安全地写入文件
            with file_lock:
                f.write(json.dumps(res) + '\n')
                f.flush()

# --- 统计指标阶段 ---
print("\nCalculating metrics...")
convert = inflect.engine()
correct = 0
total = 0

all_types = ['object category', 'object existence', 'object quantity', 'object color', 'object shape', 'object size', 'object position', 'object direction', 'image', 'scene type', 'reasoning', 'rural or urban']

correct_per_type = {k: 0 for k in all_types}
total_per_type = {k: 0 for k in all_types}
invalid_type = 0

for item in results:
    q_type = item.get('type', '').lower()
    if q_type == 'image': q_type = 'scene type'
    if q_type == 'rural or urban': q_type = 'scene type'

    if q_type in all_types:
        total_per_type[q_type] += 1
    else:
        invalid_type += 1

    if item.get('correct') == '1':
        correct += 1
        if q_type in all_types:
            correct_per_type[q_type] += 1
    total += 1

print(f'\nTotal questions: {total} | Valid types: {sum(total_per_type.values())} | Invalid types: {invalid_type}')
print(f'Overall Accuracy: {(correct/total * 100):.2f}%')
print('#####################################')
acc_list = []
for k in all_types:
    if total_per_type[k] == 0:
        continue
    acc = correct_per_type[k]/total_per_type[k] * 100
    print(f'{k.ljust(20)} accuracy: {acc:.2f}% (out of {total_per_type[k]})')
    acc_list.append(acc)

print('\nLaTeX Table Format (Types + Mean):')
print(' & '.join([f'{acc:.1f}' for acc in list(acc_list) + [np.mean(acc_list)]]))