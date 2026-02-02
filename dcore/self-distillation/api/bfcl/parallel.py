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
import requests
import copy
import time
import re
import tqdm

url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
qwen3_url = "https://poc-dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

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

def reasoning_processes_composition(model_name, text):
    messages = [
      {
          "role": "system",
          "content": """You are a language expert skilled in synthesizing reasoning processes. Now you have a task A along with a series of subtasks decomposed from this task A and the reasoning processes of these subtasks. I want you to systematically utilize this task, the subtask list, and their reasoning processes to compose an overall reasoning process about task. In this reasoning process, you should first deompose A into subtasks, then create a coherent overall reasoning process based on subtasks' reasoning processes while maintaining fluent language and preserving the original meaning.
          """
      },
      {
          "role":"user",
          "content": text
      }

    ]
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=url,)

    response = client.chat.completions.create(model=model_name, messages=messages, stream=False)
    content = response.choices[0].message.content
    return content

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

def test_openai(model_name, text):
    messages = [
        {
            "role": "system",
            "content": 
            """You are an expert at breaking down tasks based on the current toolset and the user's query. Below, you will see a very detailed description of the tools, including a large number of tools as well as their specific usage methods and scenarios. After that, you will see the user's query. This query may contain multiple subtasks. Please break down these subtasks and output them in JSON format. If you think this query cannot be broken down into subtasks, then use the query itself as the subtask. Here are some examples:\n
<example_1>
Tools:[{"name": "authenticate_twitter", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Authenticate a user with username and password.", "parameters": {"type": "dict", "properties": {"username": {"type": "string", "description": "Username of the user."}, "password": {"type": "string", "description": "Password of the user."}}, "required": ["username", "password"]}, "response": {"type": "dict", "properties": {"authentication_status": {"type": "boolean", "description": "True if authenticated, False otherwise."}}}}, {"name": "comment", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Comment on a tweet for the authenticated user.", "parameters": {"type": "dict", "properties": {"tweet_id": {"type": "integer", "description": "ID of the tweet to comment on."}, "comment_content": {"type": "string", "description": "Content of the comment."}}, "required": ["tweet_id", "comment_content"]}, "response": {"type": "dict", "properties": {"comment_status": {"type": "string", "description": "Status of the comment action."}}}}, {"name": "follow_user", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Follow a user for the authenticated user.", "parameters": {"type": "dict", "properties": {"username_to_follow": {"type": "string", "description": "Username of the user to follow."}}, "required": ["username_to_follow"]}, "response": {"type": "dict", "properties": {"follow_status": {"type": "boolean", "description": "True if followed, False if already following."}}}}, {"name": "get_tweet", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Retrieve a specific tweet.", "parameters": {"type": "dict", "properties": {"tweet_id": {"type": "integer", "description": "ID of the tweet to retrieve."}}, "required": ["tweet_id"]}, "response": {"type": "dict", "properties": {"id": {"type": "integer", "description": "ID of the retrieved tweet."}, "username": {"type": "string", "description": "Username of the tweet's author."}, "content": {"type": "string", "description": "Content of the tweet."}, "tags": {"type": "array", "description": "List of tags associated with the tweet.", "items": {"type": "string"}}, "mentions": {"type": "array", "description": "List of users mentioned in the tweet.", "items": {"type": "string"}}}}}, {"name": "get_tweet_comments", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Retrieve all comments for a specific tweet.", "parameters": {"type": "dict", "properties": {"tweet_id": {"type": "integer", "description": "ID of the tweet to retrieve comments for."}}, "required": ["tweet_id"]}, "response": {"type": "dict", "properties": {"comments": {"type": "array", "description": "List of dictionaries, each containing comment information.", "items": {"type": "dict", "properties": {"username": {"type": "string", "description": "Username of the commenter."}, "content": {"type": "string", "description": "Content of the comment."}}}}}}}, {"name": "get_user_stats", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Get statistics for a specific user.", "parameters": {"type": "dict", "properties": {"username": {"type": "string", "description": "Username of the user to get statistics for."}}, "required": ["username"]}, "response": {"type": "dict", "properties": {"tweet_count": {"type": "integer", "description": "Number of tweets posted by the user."}, "following_count": {"type": "integer", "description": "Number of users the specified user is following."}, "retweet_count": {"type": "integer", "description": "Number of retweets made by the user."}}}}, {"name": "get_user_tweets", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Retrieve all tweets from a specific user.", "parameters": {"type": "dict", "properties": {"username": {"type": "string", "description": "Username of the user whose tweets to retrieve."}}, "required": ["username"]}, "response": {"type": "dict", "properties": {"user_tweets": {"type": "array", "description": "List of dictionaries, each containing tweet information.", "items": {"type": "dict", "properties": {"id": {"type": "integer", "description": "ID of the retrieved tweet."}, "username": {"type": "string", "description": "Username of the tweet's author."}, "content": {"type": "string", "description": "Content of the tweet."}, "tags": {"type": "array", "description": "List of tags associated with the tweet.", "items": {"type": "string"}}, "mentions": {"type": "array", "description": "List of users mentioned in the tweet.", "items": {"type": "string"}}}}}}}}, {"name": "list_all_following", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: List all users that the authenticated user is following.", "parameters": {"type": "dict", "properties": {}, "required": []}, "response": {"type": "dict", "properties": {"following_list": {"type": "array", "description": "List of all users that the authenticated user is following.", "items": {"type": "string"}}}}}, {"name": "mention", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Mention specified users in a tweet.", "parameters": {"type": "dict", "properties": {"tweet_id": {"type": "integer", "description": "ID of the tweet where users are mentioned."}, "mentioned_usernames": {"type": "array", "items": {"type": "string"}, "description": "List of usernames to be mentioned."}}, "required": ["tweet_id", "mentioned_usernames"]}, "response": {"type": "dict", "properties": {"mention_status": {"type": "string", "description": "Status of the mention action."}}}}, {"name": "post_tweet", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Post a tweet for the authenticated user.", "parameters": {"type": "dict", "properties": {"content": {"type": "string", "description": "Content of the tweet."}, "tags": {"type": "array", "items": {"type": "string"}, "description": "List of tags for the tweet. Tag name should start with #. This is only relevant if the user wants to add tags to the tweet.", "default": []}, "mentions": {"type": "array", "items": {"type": "string"}, "description": "List of users mentioned in the tweet. Mention name should start with @. This is only relevant if the user wants to add mentions to the tweet.", "default": []}}, "required": ["content"]}, "response": {"type": "dict", "properties": {"id": {"type": "integer", "description": "ID of the posted tweet."}, "username": {"type": "string", "description": "Username of the poster."}, "content": {"type": "string", "description": "Content of the tweet."}, "tags": {"type": "array", "description": "List of tags associated with the tweet.", "items": {"type": "string"}}, "mentions": {"type": "array", "description": "List of users mentioned in the tweet.", "items": {"type": "string"}}}}}, {"name": "posting_get_login_status", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Get the login status of the current user.", "parameters": {"type": "dict", "properties": {}, "required": []}, "response": {"type": "dict", "properties": {"login_status": {"type": "boolean", "description": "True if the current user is logged in, False otherwise."}}}}, {"name": "retweet", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Retweet a tweet for the authenticated user.", "parameters": {"type": "dict", "properties": {"tweet_id": {"type": "integer", "description": "ID of the tweet to retweet."}}, "required": ["tweet_id"]}, "response": {"type": "dict", "properties": {"retweet_status": {"type": "string", "description": "Status of the retweet action."}}}}, {"name": "search_tweets", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Search for tweets containing a specific keyword.", "parameters": {"type": "dict", "properties": {"keyword": {"type": "string", "description": "Keyword to search for in the content of the tweets."}}, "required": ["keyword"]}, "response": {"type": "dict", "properties": {"matching_tweets": {"type": "array", "description": "List of dictionaries, each containing tweet information.", "items": {"type": "dict", "properties": {"id": {"type": "integer", "description": "ID of the retrieved tweet."}, "username": {"type": "string", "description": "Username of the tweet's author."}, "content": {"type": "string", "description": "Content of the tweet."}, "tags": {"type": "array", "description": "List of tags associated with the tweet.", "items": {"type": "string"}}, "mentions": {"type": "array", "description": "List of users mentioned in the tweet.", "items": {"type": "string"}}}}}}}}, {"name": "unfollow_user", "description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, commenting, and following users on Twitter. Tool description: Unfollow a user for the authenticated user.", "parameters": {"type": "dict", "properties": {"username_to_unfollow": {"type": "string", "description": "Username of the user to unfollow."}}, "required": ["username_to_unfollow"]}, "response": {"type": "dict", "properties": {"unfollow_status": {"type": "boolean", "description": "True if unfollowed, False if not following."}}}}, {"name": "cat", "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Display the contents of a file of any extension from currrent directory.", "parameters": {"type": "dict", "properties": {"file_name": {"type": "string", "description": "The name of the file from current directory to display. No path is allowed. "}}, "required": ["file_name"]}, "response": {"type": "dict", "properties": {"file_content": {"type": "string", "description": "The content of the file."}}}}, {"name": "cd", "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Change the current working directory to the specified folder.", "parameters": {"type": "dict", "properties": {"folder": {"type": "string", "description": "The folder of the directory to change to. You can only change one folder at a time. "}}, "required": ["folder"]}, "response": {"type": "dict", "properties": {"current_working_directory": {"type": "string", "description": "The new current working directory path."}}}}, {"name": "cp", "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Copy a file or directory from one location to another.  If the destination is a directory, the source file or directory will be copied into the destination directory.  Both source and destination must be local to the current directo]
Users: Copy all the text files of the 'Quarter1_Reports' directory and place it in a new directory naming it 'Archived_Quarter1'.
Results:
 [\n  {\n    \"step\": 1,\n    \"description\": \"Use the 'find' tool to search for all text files within the 'Quarter1_Reports' directory.\"\n  },\n  {\n    \"step\": 2,\n    \"description\": \"Create a new directory named 'Archived_Quarter1' using the 'mkdir' tool.\"\n  },\n  {\n    \"step\": 3,\n    \"description\": \"For each found text file, use the 'cp' tool to copy it from the 'Quarter1_Reports' directory to the newly created 'Archived_Quarter1' directory.\"\n  }\n]
</example_1>
"""
        },
        {
            "role": "user",
            "content": text
        }
        
      ]

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=url,)

    response = client.chat.completions.create(model=model_name, messages=messages, stream=False)
    content = response.choices[0].message.content
    return content

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

def remove_json_code_blocks(text):
    """
    检查并去除文本中的```json```代码块标记
    """
    #检查是否存在```json和```标记
    has_json_blocks = bool(re.search(r'```json|```', text))
    
    if has_json_blocks:
        print("发现了```json```代码块标记")# 去除```json开始标记和```结束标记
        cleaned_text = re.sub(r'```json\s*', '', text)
        cleaned_text = re.sub(r'```\s*', '', cleaned_text)
        print("已去除代码块标记")
        return cleaned_text.strip()
    else:
        print("未发现```json```代码块标记")
        return text

def load_result(split_result):
    split_steps = []
    # if '```json' in split_result:
    #     split_result = remove_json_code_blocks(split_result)
    # results = json.loads(split_result)
    for step in split_result:
      split_steps.append(step['description'])
    return split_steps

def extract_json_array_precise(text):
    #匹配完整的JSON数组结构
    # pattern = r'\[\s*(?:\{\s*"step":\s*\d+,\s*"description":\s*"[^"]*"\s*\}(?:\s*,\s*)?)+\s*\]'
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

def split_query(task, output_name):
    result = []
   
    # query_mt_list = load_file(task)
    with open(task, 'r') as f:
      query_mt_list = json.load(f) 
    
    i = 0
    # decompose_i = 0
    for query_mt in tqdm.tqdm(query_mt_list):
        # if i > 50:
        #     continue
        policy = query_mt['conversations'][0]['content'].split('Here is a list of functions in JSON format that you can invoke:\n')[0]
        functions = query_mt['conversations'][0]['content'].split('Here is a list of functions in JSON format that you can invoke:\n')[-1]
        functions = functions.split('Should you decide to')[0]
        query_content = query_mt['conversations'][-1]['content']
        chat_history = query_mt['conversations'][1:-1]
        gt_content = query_mt['ground_truth']
        task_content = f"Policy:{policy}\nTool List:{functions}\n Chat History:{chat_history} \n Query:{query_content}\n Tool Invocation Results:{gt_content}\n SubTasks:"
        print(task_content)
        # split_result = test_openai("qwen-max-2025-01-25", task_content)
        # split_result = test_openai_new("qwen-max-latest", task_content)
        split_result = test_openai_tau_bench("deepseek-r1-0528", task_content)
        print(split_result)
        try:
            split_result = extract_json_array_precise(split_result)
            # split_steps = load_result(split_result)
        except:
            continue
  
        query_mt['sub_tasks'] = split_result
        result.append(query_mt)
        i += 1
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=True, ensure_ascii=False)

def split_query_qwen3_coder(task, output_name):
    result = []
   
    # query_mt_list = load_file(task)
    with open(task, 'r') as f:
      query_mt_list = json.load(f) 
    
    i = 0
    # decompose_i = 0
    for query_mt in tqdm.tqdm(query_mt_list):
        # if i > 50:
        #     continue
        policy = query_mt['conversations'][0]['content'].split('Here is a list of functions in JSON format that you can invoke:\n')[0]
        functions = query_mt['conversations'][0]['content'].split('Here is a list of functions in JSON format that you can invoke:\n')[-1]
        if 'Should you decide to' in functions:
            functions = functions.split('Should you decide to')[0]
        query_content = query_mt['conversations'][-1]['content']
        chat_history = query_mt['conversations'][1:-1]
        gt_content = query_mt['ground_truth']
        task_content = f"Policy:{policy}\nTool List:{functions}\n Chat History:{chat_history} \n Query:{query_content}\n Tool Invocation Results:{gt_content}\n SubTasks:"
        # print(task_content)
        # split_result = test_openai("qwen-max-2025-01-25", task_content)
        # split_result = test_openai_new("qwen-max-latest", task_content)
        split_result = test_openai_tau_bench("qwen3-coder-480b-a35b-instruct", task_content)
        print(split_result)
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

def system_prompt_pre_processing_chat_model(query, function_docs):
    """
    Add a system prompt to the chat model to instruct the model on the available functions and the expected response format.
    If the prompts list already contains a system prompt, append the additional system prompt content to the existing system prompt.
    """
    messages = []
    # assert type(messages) == list

    system_prompt_template = DEFAULT_SYSTEM_PROMPT

    system_prompt = system_prompt_template.format(functions=function_docs)

    messages.insert(
            0,
            {"role": "system", "content": system_prompt},
        )
    messages.append({"role":"user", "content": query})

    return messages

        
def request_reasoning_model_api(messages):
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=url,)
    # response = client.chat.completions.create(model="deepseek-r1-0528", messages=messages, stream=False, extra_body={"enable_thinking": True})
    response = client.chat.completions.create(model="deepseek-r1-distill-qwen-7b", messages=messages)
    answer = response.choices[0].message.content
    reasoning_content = response.choices[0].message.reasoning_content
    return reasoning_content, answer

def request_reasoning_model_r1(messages):
    model_name = "deepseek-r1-0528"

    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=url,)
    response = client.chat.completions.create(model=model_name, messages=messages, stream=False)
    answer = response.choices[0].message.content
    reasoning_content = response.choices[0].message.reasoning_content

    print(reasoning_content, answer)
    return reasoning_content, answer

def request_reasoning_model(messages):
    payload = json.dumps({
    "model": "pre-qwen_intent_stress_test_chat",
    "input":{"messages": messages},
    "parameter":{"teperature":0.1}})
    
    headers = {
    'Authorization': os.getenv("DASHSCOPE_API_KEY"),
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", qwen3_url, headers=headers, data=payload)
    text = json.loads(response.text)
    # print(text)
    # print(text['output']['choices'][0]['message']['content'])
    try:
      content = text['output']['choices'][0]['message']['content']
      reasoning_content, answer = content.split('</think>\n\n')
    except:
      print('model response error:', text)
      time.sleep(3)
      response = requests.request("POST", qwen3_url, headers=headers, data=payload)
      text = json.loads(response.text)
      content = text['output']['choices'][0]['message']['content']
      reasoning_content, answer = content.split('</think>\n\n')
    return reasoning_content, answer

def reasoning_processes_composition_mannual(task, subtasks, reasoning_processes):
    full_reasoning = f"""First, let me analyze the overall task: {task} To complete this task, I need to break it down into the following subtasks:\n"""
     # Add subtask list
    for i, subtask in enumerate(subtasks, 1):
         full_reasoning += f"{i}. {subtask}\n"
     
    full_reasoning += "\nNow let me analyze the execution process of each subtask step by step:\n\n"
     
    # Add reasoning process for each subtask
    for i, (subtask, reasoning_process) in enumerate(zip(subtasks, reasoning_processes), 1):
        full_reasoning += f"Subtask {i}: {subtask}\n"
        # Extract reasoning thinking process (remove tags and function call parts)
        thinking_part = reasoning_process[0].replace("", "").replace("", "").strip()
        thinking_part = thinking_part.split('<think>\n')[-1]
        function_call = reasoning_process[1].strip()
        full_reasoning += f"Reasoning process: {thinking_part}\n"
        full_reasoning += f"Execute command: {function_call}\n\n"
    return full_reasoning

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

def composing_reasoning_process_offline(task_file, composed_reasoning_file):
    with open(task_file, 'r') as f:
        tasks = json.load(f)
    
    composed_reasoning = []
    composed_answer = []
    
    for task in tasks:
        question = task['conversations'][-1]['content']
        sub_tasks = task['sub_tasks']
        reasoning_steps = task['reasoning_steps']
        query = f"Task:{question}\n\n SubTasks:{sub_tasks}\n\n ReasoningProcesses:{reasoning_steps}"
        # print(query)
        # composed_reasoning_process = reasoning_processes_composition("qwen-max-latest", query)
        composed_reasoning_process = reasoning_processes_composition_mannual(question, sub_tasks, reasoning_steps)
        # print(composed_reasoning_process)
        answer_list = []
        for step in reasoning_steps:
            answer_list.append(step[1].strip())
        composed_answer = combine_bracketed_strings(answer_list)
        print(composed_answer)
        task['composed_reasoning_process'] = "<think>\n\n" + composed_reasoning_process + "</think>\n\n"
        task['composed_answer'] = composed_answer
        composed_reasoning.append(task)

    with open(composed_reasoning_file, 'w') as f:
        json.dump(composed_reasoning, f, indent=True, ensure_ascii=False)

def split_reasoning_r1(task, output_name):
    result = []
    with open(task, 'r') as f:
      query_mt_list = json.load(f) 

    for query_mt in tqdm.tqdm(query_mt_list):
        reasoning_processes = []
        new_mt = copy.deepcopy(query_mt)
        # ground_truth = ast_parse(new_mt['ground_truth'])
        # if len(ground_truth) != len(query_mt['sub_tasks']):
            # continue
        # new_mt['conversations']
        new_mt['conversations'].pop()
        tool_response_list = new_mt['tool_response'] 
        if len(tool_response_list) != len(new_mt['sub_tasks']):
            continue
        i = 0
        # try:
        if(1):
            for step in query_mt['sub_tasks']:
                new_mt['conversations'].append({'role':'user', 'content': step})
                reasoning = request_reasoning_model_r1(new_mt['conversations'])
                new_mt['conversations'].append({'role':'assistant', 'content': reasoning[-1]})
                reasoning_processes.append(reasoning)
                new_mt['conversations'].append({'role':'user', 'content': f"{tool_response_list[i]}"})
                i += 1
        # except:
        #     continue
        query_mt['reasoning_steps'] = reasoning_processes 
        result.append(query_mt)
        # i += 1
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=True, ensure_ascii=False)

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
                reasoning = request_reasoning_model(new_mt['conversations'])
                new_mt['conversations'].append({'role':'assistant', 'content': reasoning[-1]})
                # print(new_mt)
                reasoning_processes.append(reasoning)
                new_mt['conversations'].append({'role':'user', 'content': f"{tool_response_list[i]}"})
                i += 1
        except:
            continue
        query_mt['reasoning_steps'] = reasoning_processes 
        result.append(query_mt)
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=True, ensure_ascii=False)

def split_reasoning_single_step(task, output_name):
    result = []
    with open(task, 'r') as f:
      query_mt_list = json.load(f) 

    j = 0
    for query_mt in tqdm.tqdm(query_mt_list):
        if j < 4000:
            j += 1
            continue
        reasoning_processes = []
        new_mt = copy.deepcopy(query_mt)
        # ground_truth = ast_parse(new_mt['ground_truth'])
        # if len(ground_truth) != len(query_mt['sub_tasks']):
            # continue
        # new_mt['conversations']
        new_mt['conversations'].pop()
        tool_response_list = new_mt['tool_response'] 
        if len(tool_response_list) != len(new_mt['sub_tasks']):
            continue
        if len(new_mt['sub_tasks']) >= 2:
            continue
        i = 0
        try:
            for step in query_mt['sub_tasks']:
                new_mt['conversations'].append({'role':'user', 'content': step})
                # new_mt['id'] = decompose_i
                reasoning = request_reasoning_model(new_mt['conversations'])
                new_mt['conversations'].append({'role':'assistant', 'content': reasoning[-1]})
                # print(new_mt)
                reasoning_processes.append(reasoning)
                new_mt['conversations'].append({'role':'user', 'content': f"{tool_response_list[i]}"})
                i += 1
        except:
            continue
          # result.append(new_mt)
            # decompose_i += 1
        # query_mt['sub_tasks'] = split_steps
        # query_mt['tool_response'] = tool_response_list
        query_mt['reasoning_steps'] = reasoning_processes 
        result.append(query_mt)
        # i += 1
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=True, ensure_ascii=False)

def split_reasoning_multi_step(task, output_name):
    result = []
   
    with open(task, 'r') as f:
      query_mt_list = json.load(f) 

    j = 0
    for query_mt in tqdm.tqdm(query_mt_list):
        if j < 4000:
            j += 1
            continue
        reasoning_processes = []
        new_mt = copy.deepcopy(query_mt)
        # ground_truth = ast_parse(new_mt['ground_truth'])
        # if len(ground_truth) != len(query_mt['sub_tasks']):
            # continue
        # new_mt['conversations']
        new_mt['conversations'].pop()
        tool_response_list = new_mt['tool_response'] 
        if len(tool_response_list) != len(new_mt['sub_tasks']):
            continue
        if len(new_mt['sub_tasks']) < 2:
            continue
        i = 0
        try:
            for step in query_mt['sub_tasks']:
                new_mt['conversations'].append({'role':'user', 'content': step})
                # new_mt['id'] = decompose_i
                reasoning = request_reasoning_model(new_mt['conversations'])
                new_mt['conversations'].append({'role':'assistant', 'content': reasoning[-1]})
                # print(new_mt)
                reasoning_processes.append(reasoning)
                new_mt['conversations'].append({'role':'user', 'content': f"{tool_response_list[i]}"})
                i += 1
        except:
            continue
        query_mt['reasoning_steps'] = reasoning_processes 
        result.append(query_mt)
        with open(output_name, 'w') as f:
            json.dump(result, f, indent=True, ensure_ascii=False)

def reasoning_processes_composition_mannual_parallel(task, subtasks, reasoning_processes):
    full_reasoning = f"""OK, let me analyze the overall task: {task}\n To complete this task, I need to break it down into the following subtasks:\n"""
     # Add subtask list
    for i, subtask in enumerate(subtasks, 1):
         full_reasoning += f"{i}. {subtask}\n"
    
    full_reasoning += "Wait, I discovered that there are no dependencies between these subtasks, meaning that no subtask's input requires the output from another subtask's execution, so I can run these subtasks in parallel.\n"
     
    # Add reasoning process for each subtask
    for i, (subtask, reasoning_process) in enumerate(zip(subtasks, reasoning_processes), 1):
        full_reasoning +=  f"Now I should analyze the execution process of subtask {i}: {subtask}\n"
        
        thinking_part = reasoning_process[0].replace("", "").replace("", "").strip()
        thinking_part = thinking_part.split('<think>\n')[-1]
        full_reasoning += f"{thinking_part}\n"
        
    answer_list = []
    for step in reasoning_processes:
        answer_list.append(step[1].strip())
    composed_answer = combine_bracketed_strings(answer_list)
    full_reasoning = full_reasoning + "Finally, I can output a parallel tool call result, and after verification, this result meets the user's requirements:" + composed_answer
    return full_reasoning, composed_answer

def composing_reasoning_process_offline_parallel(task_file, composed_reasoning_file):
    with open(task_file, 'r') as f:
        tasks = json.load(f)
    
    train_set = []
    total_id = 0
    
    for task in tqdm.tqdm(tasks):
        task['conversations'].pop()
        question = task['conversations'][-1]['content']
        sub_tasks = task['sub_tasks']
        reasoning_steps = task['reasoning_steps']
       
        reasoning, answer = reasoning_processes_composition_mannual_parallel(question, sub_tasks, reasoning_steps)

        task['conversations'].append({'role':'assistant', 'content': f"<think>\n\n{reasoning}</think>\n\n{answer}"})
        train_set.append({'id': total_id, 'conversations': copy.deepcopy(task['conversations'])})
        total_id += 1

        with open(composed_reasoning_file, 'w') as f:
            json.dump(train_set, f, indent=True, ensure_ascii=False)


if __name__ == '__main__':
    #Implementation of Algorithm 1 in D-CORE paper for irrelevant scenario

    task_decomposition()

    reasoning_generation()

    composing_reasoning_process_offline_parallel()