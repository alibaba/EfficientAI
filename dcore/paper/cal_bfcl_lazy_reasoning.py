import json
import ast
from transformers import AutoTokenizer
import numpy as np

def sort_key(entry):
    """
    Index comes in two forms: TestCategory_Index or TestCategory_Index-FuncDocSubIndex-PromptSubIndex; both 0-indexed.

    TestCategory_Index: For example, `simple_20` means the 21st entry in the `simple` test category.

    TestCategory_Index-FuncDocSubIndex-PromptSubIndex is used when there are multiple prompts for a single function doc; this only happens in the live dataset.
    FuncDocSubIndex increments for each unique function doc.
    PromptSubIndex is per function doc. It resets to 0 for each function doc.
        For example, `live_simple_19-3-15` means the 20th entry in the `live_simple` test category.
        This entry has the 4th unique function doc and the 16th prompt for that function doc (there are at least 15 other prompts for this same function doc in this category).

    In either case, the universal index is enough to sort the entries.
    """
    parts = entry["id"].rsplit("_", 1)
    test_category, index = parts[0], parts[1]
    # This handles the case where the index is in the form TestCategory_Index-FuncDocSubIndex-PromptSubIndex
    if "-" in index:
        index = index.split("-")[0]
    return (test_category, int(index))

def load_file(file_path, sort_by_id=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))

    if sort_by_id:
        result.sort(key=sort_key)
    return result

def count_wait(text):
    text_lower = text.lower()
    count = text_lower.count("wait")
    count += text_lower.count("but")
    count += text_lower.count("alternatively")
    count += text_lower.count("however")
    return count


def parse_error_id(data):
    try:
        result = data.get('model_result_decoded', '')
        id = data.get('id', '')
        return id, result
    except json.JSONDecodeError as e:
        return None, None
    except Exception as e:
        return None, None
    
def parse_result_and_reason(data):
    try:
        result = data.get('result', '')
        reasoning_content = data.get('reasoning_content', '')
        id = data.get('id', '')
        # reasoning_content = data.get('reasoning_content', '')
        return id, result, reasoning_content
    except json.JSONDecodeError as e:
        return None, None, None
    except Exception as e:
        return None, None, None
    
def parse_result_and_reason_mt(data):
    try:
        result = data.get('result', '')
        reasoning_content = data.get('reasoning_content', '')
        id = data.get('id', '')
        return id, result, reasoning_content
    except json.JSONDecodeError as e:
        return None, None, None
    except Exception as e:
        return None, None, None

def compute_budget(task, tokenizer, model_name):
    result_think_content = f"result/{model_name}/" + task + "_result.json"
    score_file = f"score/{model_name}/" + task + "_score.json"

    score_think_content = load_file(score_file)

    wrong_id_list = []
    for c in score_think_content:
        if "total_count" in c.keys():
            continue
        id = str(c['id'])
        wrong_id_list.append(id)

    reasoning_content_list = []
    think_result_list = []

    wait_num = 0
    think_token_num_list = []
    wait_num_list = []
    valid_list = []

    with open(result_think_content, 'r', encoding='utf-8') as file:
         for line in file:
            line = line.strip()
            if line:
                if 'total_count' in line:
                    decode = ast.literal_eval(line)
                    total_count = decode['total_count']
                    # print('no think accuracy:', decode['accuracy'])
                    continue
                data = json.loads(line)
                id, result, reasoning_content = parse_result_and_reason(data)
                if id in wrong_id_list:
                #     continue
                    valid_list.append(1)
                else:
                    valid_list.append(0)
                reasoning_content_list.append(reasoning_content)
                wait_num += count_wait(reasoning_content)
                think_result_list.append(result)

    
    total_result_token = 0
    for result in think_result_list:
        total_result_token += len(tokenizer.encode(result, add_special_tokens=False))

    for reasoning in reasoning_content_list:
        total_result_token += len(tokenizer.encode(reasoning, add_special_tokens=False))
        think_token_num_list.append(len(tokenizer.encode(reasoning, add_special_tokens=False)))
        wait_num_list.append(count_wait(reasoning))

    avg_result_token = total_result_token/len(think_result_list)
    # print('avg token', avg_result_token)
    # print('avg wait num:', sum(wait_num_list)/len(think_result_list))
    return total_result_token, len(think_result_list), wait_num_list, think_token_num_list, valid_list


def compute_budget_mt(task, tokenizer, model_name):
    result_think_content = f"result/{model_name}/" + task + "_result.json"
    score_file = f"score/{model_name}/" + task + "_score.json"
    score_think_content = load_file(score_file)
    wrong_id_list = []
    for c in score_think_content:
        if "total_count" in c.keys():
            continue
        id = str(c['id'])
        wrong_id_list.append(id)

    reasoning_content_list = []
    think_result_list = []
    wait_num_list = []

    valid_list = []
    with open(result_think_content, 'r', encoding='utf-8') as file:
         for line in file:
            line = line.strip()
            if line:
                if 'total_count' in line:
                    decode = ast.literal_eval(line)
                    total_count = decode['total_count']
                    # print('no think accuracy:', decode['accuracy'])
                    continue
                data = json.loads(line)

                # print(data.keys())
                # import pdb
                # pdb.set_trace()

                id, results, reasoning_contents = parse_result_and_reason_mt(data)
                #     continue
                for result in results:
                    think_result_list.extend(result)
                for reasoning_content in reasoning_contents:
                    if id in wrong_id_list:
                        valid_list.extend([1] * len(reasoning_content))
                    else:
                        valid_list.extend([0] * len(reasoning_content))
                    reasoning_content_list.extend(reasoning_content)
    
    total_result_token = 0
    think_token_num_list = [] 
    for result in think_result_list:
        total_result_token += len(tokenizer.encode(result, add_special_tokens=False))
    for reasoning in reasoning_content_list:
        total_result_token += len(tokenizer.encode(reasoning, add_special_tokens=False))
        think_token_num_list.append(len(tokenizer.encode(reasoning, add_special_tokens=False)))
        wait_num_list.append(count_wait(reasoning))

    avg_result_token = total_result_token/len(think_result_list)

    # print('avg token', avg_result_token)
    # print('avg wait num:', sum(wait_num_list)/len(reasoning_content_list))
    
    return total_result_token, len(reasoning_content_list), wait_num_list, think_token_num_list, valid_list

def single_turn_think_and_wait(tokenizer, model_name):
    total_token = 0
    total_num = 0
    # total_wait_num = 0
    total_wait_num_list_parallel = []
    total_think_token_num_list_parallel = []
    total_wait_num_list_irr = []
    total_think_token_num_list_irr = []
    para_wrong_id_list = []
    irr_wrong_id_list = []

    token, num, wait_num_list, think_token_num_list, wrong_id_list = compute_budget('BFCL_v3_live_parallel', tokenizer, model_name)
    total_token += token
    total_num += num
    total_wait_num_list_parallel.extend(wait_num_list)
    total_think_token_num_list_parallel.extend(think_token_num_list)
    para_wrong_id_list.extend(wrong_id_list)

    token, num, wait_num_list, think_token_num_list, wrong_id_list = compute_budget('BFCL_v3_parallel', tokenizer, model_name)
    total_token += token
    total_num += num
    total_wait_num_list_parallel.extend(wait_num_list)
    total_think_token_num_list_parallel.extend(think_token_num_list)
    para_wrong_id_list.extend(wrong_id_list)

    token, num, wait_num_list, think_token_num_list, wrong_id_list  = compute_budget('BFCL_v3_live_irrelevance', tokenizer, model_name)
    total_token += token
    total_num += num
    total_wait_num_list_irr.extend(wait_num_list)
    total_think_token_num_list_irr.extend(think_token_num_list)
    irr_wrong_id_list.extend(wrong_id_list)

    token, num, wait_num_list, think_token_num_list, wrong_id_list = compute_budget('BFCL_v3_irrelevance', tokenizer, model_name)
    total_token += token
    total_num += num
    total_wait_num_list_irr.extend(wait_num_list)
    total_think_token_num_list_irr.extend(think_token_num_list)
    irr_wrong_id_list.extend(wrong_id_list)

    return total_think_token_num_list_parallel, total_wait_num_list_parallel, para_wrong_id_list, total_think_token_num_list_irr, total_wait_num_list_irr, irr_wrong_id_list 

def multi_turn_think_and_wait(tokenizer, model_name):
    total_token = 0
    total_num = 0
    total_wait_num_list = []
    total_think_token_num_list = []
    total_wrong_id_list = []

    token, num, wait_num, think_token_num_list, wrong_id_list = compute_budget_mt('BFCL_v3_multi_turn_base', tokenizer, model_name)
    total_token += token
    total_num += num
    total_wait_num_list.extend(wait_num)
    total_think_token_num_list.extend(think_token_num_list)
    total_wrong_id_list.extend(wrong_id_list)


    token, num, wait_num, think_token_num_list, wrong_id_list = compute_budget_mt('BFCL_v3_multi_turn_miss_func', tokenizer, model_name)
    total_token += token
    total_num += num
    total_wait_num_list.extend(wait_num)
    total_think_token_num_list.extend(think_token_num_list)
    total_wrong_id_list.extend(wrong_id_list)

    token, num, wait_num, think_token_num_list, wrong_id_list = compute_budget_mt('BFCL_v3_multi_turn_miss_param', tokenizer, model_name)
    total_token += token
    total_num += num
    total_wait_num_list.extend(wait_num)
    total_think_token_num_list.extend(think_token_num_list)
    total_wrong_id_list.extend(wrong_id_list)

    return total_think_token_num_list, total_wait_num_list, total_wrong_id_list


def filter_and_calculate_ratio_v2(para_wait_num_list, para_think_num_list, wrong_id_list):
    wait_array = np.array(para_wait_num_list)
    think_array = np.array(para_think_num_list)
    wrong_id_array = np.array(wrong_id_list)
    
    condition1 = wait_array >= 3
    condition2 = think_array > 200
    condition3 = wrong_id_array == 1
    combined_condition = condition1 & condition2 & condition3
    
    satisfied_count = np.sum(combined_condition)
    total_count = len(wait_array)
    
    ratio = satisfied_count / total_count if total_count > 0 else 0
    
    return satisfied_count, total_count, ratio

def report_lazy_reasoning_ratio(model_name):
    print(model_name)
    para_think_token_num_list, para_wait_token_num_list,para_wrong_id_list, irr_think_token_num_list, irr_wait_token_num_list, irr_wrong_id_list = single_turn_think_and_wait(tokenizer, model_name)

    multi_turn_think_token_num_list, multi_turn_wait_token_num_list, multi_turn_wrong_id_list = multi_turn_think_and_wait(tokenizer, model_name)

    _,_, para_lazy = filter_and_calculate_ratio_v2(para_wait_token_num_list, para_think_token_num_list, para_wrong_id_list)
    print(f'Para Lazy Reasoning Ratio: {para_lazy:.1%}')

    _,_, irr_lazy = filter_and_calculate_ratio_v2(irr_wait_token_num_list, irr_think_token_num_list, irr_wrong_id_list)
    print(f'Irr Lazy Reasoning Ratio: {irr_lazy:.1%}')

    _,_, mt_lazy = filter_and_calculate_ratio_v2(multi_turn_wait_token_num_list, multi_turn_think_token_num_list, multi_turn_wrong_id_list)
    print(f'Multi-Turn Lazy Reasoning Ratio: {mt_lazy:.1%}')

if __name__ == '__main__':
    model_path = 'Qwen_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    report_lazy_reasoning_ratio('Qwen3-8B')

    report_lazy_reasoning_ratio('D-CORE-14B')