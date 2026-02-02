# coding=utf-8 
# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import ast
import os
from openai import OpenAI
import re
import tqdm
import copy
import requests

url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
policy_url = "https://poc-dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

from system_prompt import AIRLINE_EXAMPLE, AIRLINE_EXAMPLE_2, AIRLINE_EXAMPLE_3

def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}

def ast_parse(input_str, language="Python"):
    if language == "Python":
        cleaned_input = input_str.strip("[]'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted


def test_openai_tau_bench(model_name, text):
    messages = [
        {
            "role": "system",
            "content": 
            """You are a task decomposition expert. Now you need to reverse-engineer the process of breaking down complex queries into subtasks based on the given information.
## Input Information:
### 1. System Policy:
[Insert the system policy here]
### 2. Available Tool List:
[{"name": Tool Name 1, "description":Function description}, {"name": Tool Name 2, "description": Function description}...]
### 3. Chat History:
[Insert the chat history here]
### 4. Query:
[Insert the query here]
### 5. Final Tool Invocation Results:
[Call Tool A, Call Tool B]
...
## Task Requirements:
Based on the above information, please reverse-engineer a reasonable subtask decomposition process based on Query and Chat History. Just output the subtask list in following format.
The output format must include 'step' to indicate the sequence number of the subtask, and 'index' to indicate the sequence number of the tool invocation result in the tool invocation results that corresponds to the current subtask.
Do not include information in the subtask description that does not exist in the chat history and query. For example, the GXWCPN in get_reservation_details(reservation_id='GXWCPN') should not appear in the subtask description.

## Output Format:
[{"step":1 , "description": Subtask 1, "index":[0]}, {"step": 2, "description": Subtask 2, "index":[1]}...]

<example_1>"""
+AIRLINE_EXAMPLE+
"""
</example_1>\n
<example_2>"""
+AIRLINE_EXAMPLE_2+
"""
</example_2>
<example_3>\n
"""
+AIRLINE_EXAMPLE_3+
"""
</example_3>
Please begin the analysis:
"""
        },
        {
            "role": "user",
            "content": text
        }
        
      ]
    print(messages)

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=url,)

    response = client.chat.completions.create(model=model_name, messages=messages, stream=False)
    content = response.choices[0].message.content
    return content

def load_result(split_result):
    split_steps = []
    for step in split_result:
      split_steps.append(step['description'])
    return split_steps

def extract_json_array_precise(text):
    pattern = r'\[\s*(?:\{\s*"step":\s*\d+\s*,\s*"description":\s*"[^"]*"\s*,\s*"index":\s*\[\s*(?:\d+(?:\s*,\s*\d+)*\s*)?\]\s*\}(?:\s*,\s*)?)+\s*\]'

    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        json_str = matches[0]
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            return json_str
    return None


def task_decomposition_online(task, output_name):
    result = []
    with open(task, 'r') as f:
      query_mt_list = json.load(f) 
    
    i = 0
    for query_mt in tqdm.tqdm(query_mt_list):
        policy = query_mt['conversations'][0]['content'].split('Here is a list of functions in JSON format that you can invoke:\n')[0]
        functions = query_mt['conversations'][0]['content'].split('Here is a list of functions in JSON format that you can invoke:\n')[-1]
        if 'Should you decide to' in functions:
            functions = functions.split('Should you decide to')[0]
        query_content = query_mt['conversations'][-1]['content']
        chat_history = query_mt['conversations'][1:-1]
        gt_content = query_mt['ground_truth']
        task_content = f"Policy:{policy}\nTool List:{functions}\n Chat History:{chat_history} \n Query:{query_content}\n Tool Invocation Results:{gt_content}\n SubTasks:"
       
        split_result = test_openai_tau_bench("qwen3-coder-480b-a35b-instruct", task_content)

        try:
            split_result = extract_json_array_precise(split_result)
            split_steps = load_result(split_result)
        except:
            continue
  
        query_mt['sub_tasks'] = split_steps
        result.append(query_mt)
        i += 1
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=True, ensure_ascii=False)

def request_reasoning_model_online(messages):
    payload = json.dumps({
    "model": "pre-qwen_intent_stress_test_chat",
    "input":{"messages": messages},
    "parameter":{"teperature":0.1}})
    
    headers = {
    'Authorization': os.getenv("DASHSCOPE_API_KEY"),
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", policy_url, headers=headers, data=payload)
    text = json.loads(response.text)
   
    try:
      content = text['output']['choices'][0]['message']['content']
      reasoning_content, answer = content.split('</think>\n\n')
    except:
      print('model response error:', text)
      response = requests.request("POST", policy_url, headers=headers, data=payload)
      text = json.loads(response.text)
      content = text['output']['choices'][0]['message']['content']
      reasoning_content, answer = content.split('</think>\n\n')
    return reasoning_content, answer

def reasoning_generation(task, output_name):
    result = []
    with open(task, 'r') as f:
      query_mt_list = json.load(f) 

    for query_mt in tqdm.tqdm(query_mt_list):
        reasoning_processes = []
        new_mt = copy.deepcopy(query_mt)
        new_mt['conversations'].pop()
        tool_response_list = new_mt['tool_response'] 
        if len(tool_response_list) != len(new_mt['sub_tasks']):
            continue
        i = 0
        try:
            for step in query_mt['sub_tasks']:
                new_mt['conversations'].append({'role':'user', 'content': step})
                reasoning = request_reasoning_model_online(new_mt['conversations'])
                new_mt['conversations'].append({'role':'assistant', 'content': reasoning[-1]})
                reasoning_processes.append(reasoning)
                new_mt['conversations'].append({'role':'user', 'content': f"{tool_response_list[i]}"})
                i += 1
        except:
            continue
        query_mt['reasoning_steps'] = reasoning_processes 
        result.append(query_mt)
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=True, ensure_ascii=False)

def reasoning_processes_composition_mannual(task, subtasks, reasoning_processes):
    reasoning_list = []
    answer_list = []
    full_reasoning = f"""OK, let me analyze the overall task: {task}\n"""
     
    # Add reasoning process for each subtask
    for i, (subtask, reasoning_process) in enumerate(zip(subtasks, reasoning_processes), 1):
        if i == 1: 
            full_reasoning_step = full_reasoning + f"Now I should analyze the execution process of subtask {i}: {subtask}\n"
        else: 
            full_reasoning_step = full_reasoning + f"I have already solve the subtask:\n" 
            for j in range(1, i):
                full_reasoning_step += f"{j}.{subtasks[j-1]}\n"
            full_reasoning_step += f"Now I should analyze the execution process of subtask {i}:{subtask}\n"
        # Extract reasoning thinking process (remove tags and function call parts)
        thinking_part = reasoning_process[0].replace("", "").replace("", "").strip()
        thinking_part = thinking_part.split('<think>\n')[-1]
        # function_call = reasoning_process[1].strip()
        full_reasoning_step += f"{thinking_part}\n"
        reasoning_list.append(full_reasoning_step)
        answer_list.append(reasoning_process[1])
    return reasoning_list, answer_list

def format_tool_response(text):
    """
    使用字符串操作而非正则表达式的版本
    """
    #检查 &lt;tool_response&gt; 后面是否需要添加换行符
    if '<tool_response>' in text:
        parts = text.split('<tool_response>')
        for i in range(1, len(parts)):
            if not parts[i].startswith('\n'):
                parts[i] = '\n' + parts[i]
        text = '<tool_response>'.join(parts)
    # 检查 &lt;/tool_response&gt;前面是否需要添加换行符
    if '</tool_response>' in text:
        parts = text.split('</tool_response>')
        for i in range(len(parts) - 1):
            if not parts[i].endswith('\n'):
                parts[i] = parts[i] + '\n'
        text = '</tool_response>'.join(parts)
    return text

def composing_reasoning_process_offline(task_file, composed_reasoning_file):
    with open(task_file, 'r') as f:
        tasks = json.load(f)
    
    train_set = []
    total_id = 0
    
    for task in tasks:
        if 1:
            question = task['conversations'][-1]['content']
        else:
            question = ''
            for conv in task['conversations']:
                if conv['role'] == 'user':
                    question += conv['content'] + '\n'
            print('question:', question)
        sub_tasks = task['sub_tasks']
        reasoning_steps = task['reasoning_steps']
        tool_response_list = task['tool_response']
        reasoning_list, answer_list = reasoning_processes_composition_mannual(question, sub_tasks, reasoning_steps)

        i = 0
        for reasoning, answer in zip(reasoning_list, answer_list):
            task['conversations'].append({'role':'assistant', 'content': f"<think>\n\n{reasoning}</think>\n\n{answer}"})
            train_set.append({'id': total_id, 'conversations': copy.deepcopy(task['conversations'])})
            total_id += 1
            task['conversations'][-1]['content'] = f"{answer}"
            task['conversations'].append({'role':'user', 'content': f"{tool_response_list[i]}"})
            i += 1
          
        
    for d in train_set:
            for conv in d['conversations']:
                if conv['role'] == 'user':
                    if '<tool_response>' in conv['content']:
                        conv['content'] = format_tool_response(conv['content'])
                    

    with open(composed_reasoning_file, 'w') as f:
        json.dump(train_set, f, indent=True, ensure_ascii=False)

def combine_bracketed_strings(string_list):
    """
    使用正则表达式提取方括号内容并组合
    """
    inner_contents = []
    pattern = r'\[(.*)\]'  # 匹配方括号内的内容
    
    for s in string_list:
        match = re.search(pattern, s.strip())
        if match:
            inner_contents.append(match.group(1))
    return "[" + ", ".join(inner_contents) + "]"

def combine_gt(string_list):
    """
    使用正则表达式提取方括号内容并组合
    """
    inner_contents = []
    for s in string_list:
        inner_contents.append(s)
    return "[" + ", ".join(inner_contents) + "]"

def text_normalization_tool_use(think_answer, input_gt):
    if isinstance(input_gt, str) is not True:
        gt = str(input_gt)
    else:
        gt = copy.copy(input_gt)
    think_answer = think_answer.strip()

    if "parameters" in think_answer:
        think_answer = think_answer.replace("parameters", "arguments")
    if gt.startswith('<tool_call>'):
        think_matches = re.findall(r"<tool_call>(.*?)</tool_call>", think_answer, re.DOTALL)
        if len(think_matches) == 0:
            think_answer = think_answer
        else:
            try:
                think_answer_tmp = think_matches[0]
                if think_answer_tmp.endswith(']') is not True:
                    think_answer_tmp += ']'
                if think_answer_tmp.startswith('[') is not True:
                    think_answer_tmp = '[' + think_answer_tmp
                
                think_answer = ast.literal_eval(think_answer_tmp)
            except:
                think_answer = think_answer

        gt_matches = re.findall(r"<tool_call>(.*?)</tool_call>", gt, re.DOTALL)
        if len(gt_matches) == 0:
            output_gt = gt 
        else:
            try:
                output_gt = gt_matches[0]
                if output_gt.endswith(']') is not True:
                    output_gt+= ']'
                if output_gt.startswith('[') is not True:
                    output_gt = '[' + output_gt
                output_gt = ast.literal_eval(output_gt)
            except:
                output_gt = gt
        return think_answer, output_gt
    elif gt.startswith('['):
        if think_answer.endswith(']') is not True:
            think_answer += ']'
        if think_answer.startswith('[') is not True:
            think_answer = '[' + think_answer
        try:
            think_answer = ast_parse(think_answer)
        except:
            try:
                think_answer = ast.literal_eval(think_answer)
            except:
                print('think answer decode failed:', think_answer)
                think_answer = think_answer
        try:
            output_gt = ast_parse(gt)
        except:
            try:
                output_gt = ast.literal_eval(gt)
            except:
                print('ground truth decode failed:', gt)
                output_gt = gt

        return think_answer, output_gt
    else:
        return think_answer, gt

def check_sub_task_and_tool_use(d_core_file, d_core_refine_file):
    with open(d_core_file,'r') as f:
        dst_data = json.load(f)
    
    accuracy_list = []
    d_core_refine_list = []

    for dst_conv in dst_data:
            tool_response_list = dst_conv['tool_response']
            reasoning_steps = dst_conv['reasoning_steps']
            answer_list = []
            for step in reasoning_steps:
                answer_list.append(step[1].strip())
            composed_answer = combine_bracketed_strings(answer_list)
            composed_gt = combine_gt(dst_conv['ground_truth'])
            answer, gt = text_normalization_tool_use(composed_answer, composed_gt)
            if answer == gt:
                accuracy_list.append(1)
                dst_conv['tool_response'] = tool_response_list
                d_core_refine_list.append(dst_conv)
            else:
                print(dst_conv['id'])
                print('sub task numbers:', len(dst_conv['sub_tasks']))
                print('ground_truth:', dst_conv['ground_truth'])
                print('composed_answer:', composed_answer)
                accuracy_list.append(0)

    print('Accuracy:', sum(accuracy_list)/len(accuracy_list))


    with open(d_core_refine_file, 'w') as f:
       json.dump(d_core_refine_list, f, indent=True, ensure_ascii=False)


if __name__ == '__main__':

    #Implementation of Algorithm 1 in D-CORE paper for sequantial scenario
    
    task_decomposition_online("paper/sample.json", "paper/decompose.json")

    reasoning_generation("paper/decompose.json", "paper/sub_task_and_reasoning.json")

    check_sub_task_and_tool_use('paper/sub_task_and_reasoning.json', 'paper/sub_task_and_reasoning_refine.json')

    composing_reasoning_process_offline("paper/sub_task_and_reasoning_refine.json", "paper/d_core_train.json")