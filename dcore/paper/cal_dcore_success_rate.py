import json
import ast
import copy
import re

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

def check_multi_tool_call(text):
    start_count = text.count('<tool_call>')
    end_count = text.count('</tool_call>')
    return min(start_count, end_count) > 1

def has_function_call(content):
    import re
    pattern = r'\[.*?\]'
    return bool(re.search(pattern, content))

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
                # print('think answer decode failed:', think_answer)
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
                # print('ground truth decode failed:', gt)
                output_gt = gt

        return think_answer, output_gt
    else:
        return think_answer, gt

def combine_bracketed_strings(string_list):
    inner_contents = []
    pattern = r'\[(.*)\]'  # 匹配方括号内的内容
    
    for s in string_list:
        match = re.search(pattern, s.strip())
        if match:
            inner_contents.append(match.group(1))
    return "[" + ", ".join(inner_contents) + "]"

def combine_gt(string_list):
    inner_contents = []
    # pattern = r'\[(.*)\]'  # 匹配方括号内的内容
    
    for s in string_list:
        # match = re.search(pattern, s.strip())
        # if match:
        inner_contents.append(s)
    return "[" + ", ".join(inner_contents) + "]"

if __name__ == '__main__':
    d_core_file = 'task_decompose_sample.json'
    
    with open(d_core_file,'r') as f:
        dst_data = json.load(f)

    accuracy_list = []
    valid_list = []
    d_core_refine_list = []
    d_core_irr_refine_list = []
    train_set = []

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
                accuracy_list.append(0)

    print('Valid Sample Number:', sum(accuracy_list))
    print('Total Sample Number:', len(accuracy_list))
    print('Success Rate:', sum(accuracy_list)/len(accuracy_list))
