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

import re
import ast
import json
import copy
import os
from collections import Counter


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0
    
    if os.getenv("REFINEDREWARD", 0) == "1":
        print("REFINEDREWARD is set to 1, so strict match is used")
        if list1 != list2:
            return 0.0
    
    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0
    

# custoimzed reward functions: format
def customize_format_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    # schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0
    
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    
    print("\n======= Answer ======= ")
    print(answer[0])
    print("\n======= Responses ======= ")
    for idx, response in enumerate(responses):
        print(f"*** Response {idx+1}***\n{response}")

    for response, ans in zip(responses, answer):
        reward = min_possible_reward
        if "<response>" in ans and "<tool_call>" not in ans:
            pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        elif "<response>" not in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$" 
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
                reward = max_possible_reward
        elif "<response>" in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1 and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward
        else:
            pattern = r"^<think>.*?</think>$"
            if re.search(pattern, response, re.DOTALL):
                reward = max_possible_reward
        
        rewards.append(reward)
        
    print("\n======= Reward for <format> =======")
    print("Reward function for <format> is called ...")
    print(rewards)
    return rewards


def customize_format_reward_func_bfcl(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    # schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0
    
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    
    print("\n======= Answer ======= ")
    print(answer[0])
    print("\n======= Responses ======= ")
    for idx, response in enumerate(responses):
        print(f"*** Response {idx+1}***\n{response}")

    for response, ans in zip(responses, answer):
        reward = min_possible_reward
        # if "<response>" in ans and "<tool_call>" not in ans:
        #     pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
        #     if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
        #         reward = max_possible_reward
        # elif "<response>" not in ans and "<tool_call>" in ans:
        #     pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$" 
        #     if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
        #         reward = max_possible_reward
        # elif "<response>" in ans and "<tool_call>" in ans:
        #     pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
        #     if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1 and response.count("<response>") == 1 and response.count("</response>") == 1:
        #         reward = max_possible_reward
        # else:
        #     pattern = r"^<think>.*?</think>$"
        #     if re.search(pattern, response, re.DOTALL):
        #         reward = max_possible_reward
        
        # 目前统一把格式转为bfcl的py格式，存在以下两种情况的格式，我们只校验<think>相关格式是否正确：
        # 1. tool_call的情况，<think>\nxxx\n</think>\n\n[]
        # 2. irrelevance的情况，<think>\nxxx\n</think>\n\nxxxx
        
        pattern = r"^<think>\n.*?\n</think>\n\n"
        if re.search(pattern, response, re.DOTALL):
            reward = max_possible_reward
        
        rewards.append(reward)
        
    print("\n======= Reward for <format> =======")
    print("Reward function for <format> is called ...")
    print(rewards)
    return rewards


# customized reward functions: length
def customize_length_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    # schedule length
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        print("SCHEDULELENGTH is set to 1, so schedule max reward for length is used")
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512
    
    """Reward function that gives higher scores to longer completions."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        if "<think>" not in response or "</think>" not in response:
            rewards.append(min_possible_reward)
            continue
        think_responses = response.split("<think>")[-1].split("</think>")[0].strip()
        reward = round(len(think_responses.split()) / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0
        
        final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        rewards.append(final_reward)
    
    print("\n======= Reward for <length> =======")
    print("Reward function for <length> is called ...")
    print(rewards)
    return rewards
                

def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    if gt_tools == pd_tools:
        print("Max possible score:", "Exact Match!")
        print("Score:", max_possible_reward)
        return max_possible_reward
    
    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if gt_tools != pd_tools:
            return min_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))
    
    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        
        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)
        
        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            
            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue
            
            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            
            # Calculate correctness score for parameter values
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    print()
    print("Max possible score:", local_max_possible)
    print("Score:", score)
    
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


def compute_tool_call_reward_bfcl(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    if gt_tools == pd_tools:
        print("Max possible score:", "Exact Match!")
        print("Score:", max_possible_reward)
        return max_possible_reward
    
    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if gt_tools != pd_tools:
            return min_possible_reward

    gt_tool_names = []
    pd_tool_names = []
    
    for tool in gt_tools:
        gt_tool_names.append(list(tool.keys())[0])
    
    for tool in pd_tools:
        pd_tool_names.append(list(tool.keys())[0])
       
        
    # gt_names = [tool["name"] for tool in gt_tools]
    # pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(gt_tool_names, pd_tool_names)
    
    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_tool_name = list(gt_tool.keys())[0]
        gt_params = list(gt_tool.values())[0]
        
        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)
        
        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            pd_tool_name = list(pd_tool.keys())[0]
            if i in used_pd_indices or pd_tool_name != gt_tool_name:
                continue
            
            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue
            
            pd_params = list(pd_tool.values())[0]
            param_name_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            
            # Calculate correctness score for parameter values
            param_value_correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_name_score + param_value_correctness_score
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    print("Max possible score:", local_max_possible)
    print("Score:", score)
    
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


# custoimzed reward functions: tool call correctness
def customize_correctness_reward_tool(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0
    
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        reward = 0.0
        
        if "<tool_call>" not in ans:
            # if "<tool_call>" not in response and "</tool_call>" not in response:
            #     reward = max_possible_reward
            # else:
            #     reward = min_possible_reward
            rewards.append(reward)
            continue

        gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        gt_tools = gt_tool_call.split("\n")
        gt_tools = [json.loads(tool) for tool in gt_tools] # each diction contains "name" and "parameter"
        
        try:
            # Change here as a constrint in training: if the format is not correct, directly give the lowest possible score
            assert "<tool_call>" in response
            assert "</tool_call>" in response
            pd_tools = response.split("<tool_call>")[1].split("</tool_call>")[0].strip().split("\n")
            pd_tools = [json.loads(tool) for tool in pd_tools]
            reward = compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward) # top reward is 2
        except:
            reward = min_possible_reward
        
        rewards.append(reward)
    
    print("\n======= Reward for <tool call> =======")
    print("Reward function for <tool call> correctness is called ...")
    print(rewards)
    return rewards


def customize_correctness_reward_tool_bfcl(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward
    
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0
    
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, ans in zip(responses, answer):
        reward = 0.0
        response_content = response.split("</think>")[-1].strip()
        ans_content = ans.split("</think>")[-1].strip()
        # pred包含tool_call
        if response_content and (response_content[0] == "[" and response_content[-1] == "]"):
            if ans_content[0] != "[" or ans_content[-1] != "]":
                reward = min_possible_reward
            
            # 两个都是tool_call，解析bfcl格式的tool_call
            try:
                pd_tools, gt_tools = text_normalization_tool_use(response_content, ans_content)
                reward = compute_tool_call_reward_bfcl(gt_tools, pd_tools, max_possible_reward, min_possible_reward)
            except:
                reward = min_possible_reward
        # irrelevance的情况
        else:
            if ans_content[0] == "[" or ans_content[-1] == "]":
                reward = min_possible_reward
            else:
                reward = max_possible_reward
        
        rewards.append(reward)
    
    print("\n======= Reward for <tool call> =======")
    print("Reward function for <tool call> correctness is called ...")
    print(rewards)
    return rewards


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


def text_normalization_tool_use(think_answer, input_gt):
    if isinstance(input_gt, str) is not True:
        gt = str(input_gt)
    else:
        gt = copy.copy(input_gt)
    think_answer = think_answer.strip()
    # gt = gt.strip()
    # print('debug:', gt)
    # if check_multi_tool_call(think_answer):
        # think_answer = merge_tool_calls(think_answer)

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
            # think_answer = ast.literal_eval(think_answer)
        except:
            try:
                think_answer = ast.literal_eval(think_answer)
            except:
                print('think answer decode failed:', think_answer)
                think_answer = think_answer
                # import pdb
                # pdb.set_trace()
        try:
            output_gt = ast_parse(gt)
            # gt = ast.literal_eval(gt)
        except:
            try:
                output_gt = ast.literal_eval(gt)
            except:
                print('ground truth decode failed:', gt)
                output_gt = gt

        return think_answer, output_gt
    else:
        return think_answer, gt


def compute_score(solution_str, ground_truth, step=0):
    exp_name = str(os.getenv("EXPERIMENT_NAME", "")).lower()
    if "llama" in exp_name:
        predict_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        # raise NotImplementedError(f"Unknown model name: {exp_name}")
        predict_str = solution_str
    
    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        print("CORRECTMAX1 is set to 1, so max score is set to 1")
        tool_max_possible = 1.0
        tool_min_possible = -1.0
    else:
        tool_max_possible = 3.0
        tool_min_possible = -3.0
    
    format_max_possible = 1.0
    format_min_possible = 0.0
    
    length_max_possible = 1.0
    length_min_possible = 0.0
    
    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]
    
    format_score = customize_format_reward_func_bfcl(completions, answer, step, format_max_possible, format_min_possible)[0]
    correctness_score = customize_correctness_reward_tool_bfcl(completions, answer, step, tool_max_possible, tool_min_possible)[0]
    
    if str(os.getenv("WITHLENGTH", 0)) == "1":
        print("WITHLENGTH is set to 1, so length score is set!")
        length_score = customize_length_reward_func(completions, answer, step, length_max_possible, length_min_possible)[0]
    else:
        length_score = 0
    
    score = format_score + correctness_score + length_score
    
    reflection_words = ["wait", "but", "however", "alternatively", "perhaps", "because"]
    lower_predict_str = predict_str.lower()
    
    reflection_count = sum(lower_predict_str.count(word) for word in reflection_words)
    
    return {
        "score": score,
        "format_score": format_score,
        "correctness_score": correctness_score,
        "length_score": length_score,
        "reflection_count": reflection_count
    }

if __name__ == "__main__":
    completions = [[
        {
            "role": "assistant",
            "content": "<think>\n\nOK, let me analyze the overall task: I am alex. Check if the current directory is under my name and list all the visible and hidden contents in the current directory now, please.\n To complete this task, I need to break it down into the following subtasks:\n1. Use the 'ls' tool with the 'a' parameter set to True to list all visible and hidden contents in the current directory.\nNow I should analyze the execution process of subtask 1: Use the 'ls' tool with the 'a' parameter set to True to list all visible and hidden contents in the current directory.\nOkay, the user wants me to use the 'ls' tool with the 'a' parameter set to True. Let me check the available functions.\n\nLooking at the 'ls' function description, it has a parameter 'a' which is a boolean. The default is False, but the user wants it set to True to show hidden files. The parameters required for 'ls' don't have any mandatory fields, but the 'a' parameter is optional. So, I need to call ls with a=True.\n\nThe function call should be [ls(a=True)]. That should list all contents, including hidden ones. I don't see any other parameters needed here. The response will give the current directory content as an array. Alright, that's the correct call.\n</think>\n\nhello"
        }
    ]]
    answer = ["<think>\nHowever, we should stop war! But nobody lisen to us.\n</think>\n\n[get_product_details(product_id=\"4896585277\")]"]
    step = 0
    max_possible_reward = 3.0
    min_possible_reward = -3.0
    # customize_correctness_reward_tool_bfcl(completions, answer, step, max_possible_reward, min_possible_reward)
    solution_str = completions[0][0]["content"]
    ground_truth = answer[0]
    compute_score(solution_str, ground_truth)
    pass
