# Copyright (c) 2021, Alibaba Cloud and its affiliates;
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess D-CORE dataset to parquet format
"""

import os
import json
import yaml
import random
import numpy as np
import pandas as pd

import argparse

def init_config(input_yaml):
    with open(input_yaml, 'r') as f:
        config = yaml.safe_load(f) 
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-yaml", type=str)
    args = parser.parse_args()
    
    config = init_config(args.input_yaml)
    output_data_parquet_path = config['output']
    split = config['split']
    all_data_path_list = config['data']
    
    
    # load dataset
    dataset = []
    for input_json_path in all_data_path_list:
        with open(input_json_path, 'r') as input_file:
            # for line in input_file:
            #     dataset.append(json.loads(line))
            dataset.extend(json.load(input_file))
    
    random.shuffle(dataset)
    
    data_source = "d_core"
    
    # function to process each example
    def process_fn(example):
        conversations = example['conversations']
        
        if conversations[-1]['role'] != 'assistant':
            conversations.append({"role": "assistant", "content": example['test_output']})
        
        if conversations[-1]['content'] == "[]":
            conversations[-1]['content'] = "None tool can use to solve this problem"
        
        data = {
            "data_source": data_source,
            "prompt": conversations[:-1],
            "ability": "tool-call",
            "reward_model": {
                "style": "rule",
                "ground_truth": conversations[-1]['content']
            },
            "extra_info": {
                "split": split,
                "index": example['id'],
                "conversations": conversations
            }
        }
        return data
    
    # process dataset using list comprehension
    processed_dataset = [process_fn(d) for d in dataset]
    
    # convert to pandas dataframe
    dataset_df = pd.DataFrame(processed_dataset)
    
    # save as parquet
    dataset_df.to_parquet(output_data_parquet_path)
    print(f"Saved datasets to {output_data_parquet_path}")
    
    # save as json for verify
    # dataset_df.to_json(output_data_parquet_path.replace(".parquet", ".json"), indent=4)