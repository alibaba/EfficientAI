# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import ast
import json
import copy
import os
from collections import Counter

def compute_score(solution_str, ground_truth, step=0):
    """The scoring function for tongyi_fraud_detection.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    exp_name = str(os.getenv("EXPERIMENT_NAME", "")).lower()
    if "llama" in exp_name:
        predict_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        # raise NotImplementedError(f"Unknown model name: {exp_name}")
        predict_str = solution_str
    
    score = 0.0
    try:
        gt_events = json.loads(ground_truth)
        gt_events_map = {}
        for event in gt_events:
            gt_events_map[event["EventType"]] = event['EventData']
        
        pred_events = json.loads(predict_str)
        
        pred_events_map = {
            "Schedule": {},
            "AddContact": {},
            "Todo": {},
            "Summary": {}
        }
        for event in pred_events:
            if event['EventType'] not in pred_events_map:
                score = 0.0
                break
            
            if pred_events_map[event['EventType']] != {}:
                score = 0.0
                break
            else:
                pred_events_map[event['EventType']] = event['EventData']
            
            if event['EventType'] == 'AddContact':
                if event['EventData']['PhoneNumber'] == gt_events_map['AddContact']['PhoneNumber']:
                    score += 1.0
            # elif gt_events_map[event["EventType"]] != {}:
            elif event['EventType'] in gt_events_map:
                score += 1.0
            else:
                score -= 1.0
        
        if pred_events_map['Summary'] == []:
            score = 0.0
    except:
        score = 0.0
    
    if score < 0:
        score = 0.0
    
    return score

if __name__ == "__main__":
    pred_str = "[{\"EventType\": \"Todo\", \"EventData\": {\"TodoList\": [\"查看京东官方网站或APP了解具体的客服电话\", \"打电话给京东客服，简单介绍自己，说明问题和需求\"]}}, {\"EventType\": \"AddContact\", \"EventData\": {\"Name\": \"京东客服\", \"PhoneNumber\": \"400-606-5500\"}}, {\"EventType\": \"Summary\", \"EventData\": {\"Description\": \"我打算联系京东的客服解决购买商品的问题，询问了客服电话，并得到了一些沟通技巧的建议。\"}}]"

    ground_truth = "[{\"EventType\": \"AddContact\", \"EventData\": {\"Name\": \"京东客服\", \"PhoneNumber\": \"400-606-5500\"}}, {\"EventType\": \"Summary\", \"EventData\": {\"Description\": \"我打算联系京东的客服解决购买商品的问题，询问了客服电话，并得到了一些沟通技巧的建议。\"}}]"

    score = compute_score(pred_str, ground_truth)
    print(score)
    pass
