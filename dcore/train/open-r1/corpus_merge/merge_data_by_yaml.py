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

import json
import yaml
import random
import argparse
import transformers

from tqdm import tqdm
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-yaml", type=Path, help="Path to the input json list")
   
    args = parser.parse_args()
    return args

def token_concat(merged_input_data, tokenizer, max_token_len):
    roles = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}
    im_start_token = tokenizer.im_start_id if hasattr(tokenizer, "im_start_id") else tokenizer.encode("<|im_start|>")
    im_end_token = tokenizer.im_end_id if hasattr(tokenizer, "im_end_id") else tokenizer.encode("<|im_end|>")
    nl_tokens = tokenizer.encode("\n")
    tmp_ids_length = 0
    merged_output_data = []
    too_long_data_num = 0
    
    for line in tqdm(merged_input_data):
        ids = []
        role_list = []
        for turn in line["conversations"]:
            role, content = roles[turn["from"]], turn["value"]
            role_list.append(role)
            ids.append(im_start_token)
            ids.extend(tokenizer.encode(role))
            ids.extend(nl_tokens)
            ids.extend(tokenizer.encode(content))
            ids.append(im_end_token)
            ids.extend(nl_tokens)

        if len(ids) >= max_token_len:
            too_long_data_num += 1
            continue
        else:
            if tmp_ids_length == 0:
                line['single_conv_last_idx'] = [len(line['conversations']) - 1]
                merged_output_data.append(line)
                tmp_ids_length = len(ids)
            else:
                if tmp_ids_length + len(ids) <= max_token_len:
                    cur_conv_length = len(merged_output_data[-1]['conversations'])
                    for conv in line['conversations']:
                        merged_output_data[-1]['conversations'].append(conv)
                    tmp_ids_length += len(ids)
                    merged_output_data[-1]['single_conv_last_idx'].append(len(merged_output_data[-1]['conversations']) - 1)
                    if 'reason_first' in line and line['reason_first']:
                        merged_output_data[-1]['reason_first'].append(cur_conv_length + line['reason_first'][0])
                else:
                    line['single_conv_last_idx'] = [len(line['conversations']) - 1]
                    merged_output_data.append(line)
                    tmp_ids_length = len(ids)
    print("too_long_data_num: ", too_long_data_num)
    return merged_output_data

def reorder_mixture_of_instruction(merged_output_data):
    assistant = 0
    programer = 0
    mathematics = 0
    for line in tqdm(merged_output_data):
        if 'assistant' not in line['conversations'][0]['value']:
            i = 0
            for data in line['conversations']:
                if data['from'] == 'system':
                    instruct_start = i
                    break
                else:
                    i += 1
            i = 1
            instruct_length = -1
            for data in line['conversations'][instruct_start+1:]:
                if data['from'] == 'system':
                    instruct_length= i
                    break
                else:
                    i += 1
            if instruct_length == -1:
                pass
            else:
                line['conversations'] = line['conversations'][instruct_start: instruct_start+instruct_length] + line['conversations'][:instruct_start] + line['conversations'][instruct_start+instruct_length:]

    for line in tqdm(merged_output_data):
        if 'assistant'in line['conversations'][0]['value']:
            assistant += 1
        if 'programer'in line['conversations'][0]['value']:
            programer += 1
        if 'mathematics'in line['conversations'][0]['value']:
            mathematics += 1
        
    total = assistant + programer + mathematics
    print(assistant, assistant/total)
    print(programer, programer/total)
    print(mathematics, mathematics/total)

    return merged_output_data

def check_system_prompt(merged_output_data):
    assistant = 0
    programmer = 0
    mathematics = 0
    Agent = 0
    for line in tqdm(merged_output_data):
        if 'assistant'in line['conversations'][0]['value']:
            assistant += 1
        if 'programmer'in line['conversations'][0]['value']:
            programmer += 1
        if 'mathematics'in line['conversations'][0]['value']:
            mathematics += 1
        if 'Agent'in line['conversations'][0]['value']:
            Agent += 1
            
    total = assistant + programmer + mathematics + Agent
    print(assistant, assistant/total)
    print(programmer, programmer/total)
    print(mathematics, mathematics/total)
    print(Agent, Agent/total)

    return merged_output_data

def convert_from_to_role(merged_output_data):
    roles_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    for line in tqdm(merged_output_data):
        new_conversations = []
        for conv in line['conversations']:
            new_conversations.append({"role": roles_map[conv['from']], "content": conv['value']})
        line['conversations'] = new_conversations

def convert_role_to_from(merged_output_data):
    roles_map = {"system": "system", "user": "human", "assistant": "gpt", "tool": "tool"}
    for line in tqdm(merged_output_data):
        new_conversations = []
        for conv in line['conversations']:
            new_conversations.append({"from": roles_map[conv['role']], "value": conv['content']})
        line['conversations'] = new_conversations

def merge_json_moi(train_data_dict, output_jsonl_file, tokenizer, max_token_len=8192):
    merged_output_data = []
    train_data_list = []

    for role in train_data_dict.keys():
        train_data_list.append(train_data_dict[role])

    train_data = merge_lists(train_data_list)
    merged_output_data = token_concat(train_data, tokenizer, max_token_len)

    merged_output_data = reorder_mixture_of_instruction(merged_output_data)

    check_system_prompt(merged_output_data)
    convert_from_to_role(merged_output_data)

    with open(output_jsonl_file, 'w', encoding='utf-8') as outfile:
        for data in merged_output_data:
            json_str = json.dumps(data, ensure_ascii=False)
            outfile.write(json_str + '\n')

def merge_json_moi_shuffle_each_system(train_data_dict, output_jsonl_file, tokenizer, max_token_len=8192):
    merged_output_data = []
    train_data_list = []

    for role in train_data_dict.keys():
        random.shuffle(train_data_dict[role])
        train_data_list.append(train_data_dict[role])

    train_data = merge_lists(train_data_list)
    merged_output_data = token_concat(train_data, tokenizer, max_token_len)

    convert_from_to_role(merged_output_data)
        
    with open(output_jsonl_file, 'w', encoding='utf-8') as outfile:
        for data in merged_output_data:
            json_str = json.dumps(data, ensure_ascii=False)
            outfile.write(json_str + '\n')

def moi_seq(train_data_dict, output_jsonl_file, tokenizer, max_token_len=8192):
    merged_output_data = []
    train_data_list = []

    for role in train_data_dict.keys():
        random.shuffle(train_data_dict[role])
        train_data_list.append(train_data_dict[role])

    merged_output_data = merge_lists(train_data_list)

    check_system_prompt(merged_output_data)
    convert_from_to_role(merged_output_data)
   
    with open(output_jsonl_file, 'w', encoding='utf-8') as outfile:
        for data in merged_output_data:
            json_str = json.dumps(data, ensure_ascii=False)
            outfile.write(json_str + '\n')

def merge_lists(lists):
    merged = []
    while any(lists):
        # 取每个列表的第一个元素（如果存在）并加入结果
        round_elements = []
        for lst in lists:
            if lst:
                round_elements.append(lst.pop(0))
        
        merged.extend(round_elements)

    return merged

def balanced_sampling(train_data_dict):
    train_data_list = []
    train_data = []

    for role in train_data_dict.keys():
        train_data_list.append(train_data_dict[role])

    train_data = merge_lists(train_data_list)

    return train_data

def reorder(line, system_key_words):
    instruct_length = 0
    instruct_start = 0
    if system_key_words not in line['conversations'][0]['value']:
        i = 0
        for data in line['conversations']:
            if data['from'] == 'system' and system_key_words in data['value']:
                instruct_start = i
                break
            else:
                i += 1
        j = 1
        for data in line['conversations'][instruct_start+1:]:
            if data['from'] == 'system':
                instruct_length= j
                break
            else:
                j += 1
        print('before:', line['conversations'])
        line['conversations'] = line['conversations'][instruct_start: instruct_start+instruct_length] + line['conversations'][:instruct_start] + line['conversations'][instruct_start+instruct_length:]
        print('after:', line['conversations'])
    return line

def json_pre_load_per_task(json_list, system_prompt, weight, sample_num=-1, default_system_prompt=['You are a helpful assistant.', 'You are an helpfull assistant.', 'You are a programmer.', 'You are a programer', 'You are an mathematics expert.', 'You are an Agent with a lot of tools.', 'You are a programer.', 'agent with a lot of helpful tools.', 'You are an Agent with a lot of tools.']):
    merged_data = []
    j = 0
    for input_json in json_list:
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if "role" in list(data[0]['conversations'][0].keys()):
                    convert_role_to_from(data)
                    
                for line in data:
                    line['id'] = str(line['id'])
                    if line["conversations"][0]['from'] == 'system':
                        if system_prompt != '': #if system prompt is not setted in config, don't chage it
                            if line["conversations"][0]['value'] in default_system_prompt:
                                line["conversations"][0]['value'] = system_prompt
                    elif line["conversations"][0]['from'] != 'system':
                        if system_prompt != '': #if system prompt is not setted in config, don't chage it
                            line["conversations"].insert(0, {'from':'system', 'value':system_prompt})
                        else:
                            line["conversations"].insert(0, {'from':'system', 'value':'You are a helpful assistant.'})
                    else:
                        j += 1
                        print(j, line["conversations"][0]['value'])
                    for conv in line["conversations"]:
                        if conv['from'] == 'human':
                            conv["weight"] = 0
                        elif conv['from'] == 'system':
                            conv["weight"] = 0
                        elif conv['from'] == 'function':
                            conv["weight"] = 0
                        elif conv['from'] == 'gpt':
                            conv["weight"] = weight
                        elif conv['from'] == 'tags':
                            conv["weight"] = weight
                        elif conv['from'] == 'slots':
                            conv["weight"] = weight
                        elif conv['from'] == 'tools':
                            conv["weight"] = 0 
                    merged_data.append(line)
        except Exception as e:
            print(e)
            print(input_json)
            
    if sample_num != -1:
        if len(merged_data) > sample_num:
            merged_data = random.sample(merged_data, sample_num)

    return merged_data

def get_annotation_list(config, sub_task_name):
    input_json_list  = []
    for json_file in config['input'][sub_task_name]['data']:
        input_json_list.append(json_file)
    return input_json_list

def get_system_prompt(config, sub_task_name):
    return config['input'][sub_task_name]['system_prompt']

def get_task_weight(config, sub_task_name):
    return config['input'][sub_task_name]['weight']

def get_roles(config):
    roles = []
    for role in config['input'].keys():
        roles.append(role)
    return roles

def init_config(input_yaml):
    with open(input_yaml, 'r') as f:
        config = yaml.safe_load(f) 
    return config

if __name__ == '__main__':
    args = get_args()
    config = init_config(args.input_yaml)
    output_jsonl_file = config['output']
    max_token_len = config['max_token_len']
    tokenizer_path = config['tokenizer_path']
    sample_num = config['sample_num']
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=max_token_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    roles = get_roles(config)

    roles_dict = {}
    train_data_dict = {}

    for role in roles:
        json_list  = get_annotation_list(config, role)
        system_prompt = get_system_prompt(config, role)
        weight = get_task_weight(config, role)
        train_data_dict[role] = json_pre_load_per_task(json_list, system_prompt, weight, sample_num)


    format = config['format']
    if format == 'moi':
        merge_json_moi(train_data_dict, output_jsonl_file, tokenizer, max_token_len)
    elif format == "moi_shuffle_each_system":
        merge_json_moi_shuffle_each_system(train_data_dict, output_jsonl_file, tokenizer, max_token_len)
    elif format == 'moi_seq':
        moi_seq(train_data_dict, output_jsonl_file, tokenizer, max_token_len)
    else:
        print('Not Implement')

    print('='* 200)
    print('output_jsonl_file:', output_jsonl_file)
    print('='* 200)
