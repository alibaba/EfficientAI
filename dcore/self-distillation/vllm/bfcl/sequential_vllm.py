import argparse
import json
import multiprocessing as mp
import os
import time
import re
import ast
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
from vllm import LLM, SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to generate text using vLLM with task decomposition")

    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-8B"
    )
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="generated/iter1")
    parser.add_argument(
        "--num_data_frac", type=int, default=0, help="Number of Data fraction"
    )
    parser.add_argument(
        "--tp_per_worker",
        type=int,
        default=1,
        help="Number of GPUs to be used per Worker",
    )
    parser.add_argument("--input_dir", type=str, default="UCLA-AGI/SPIN_iter0")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--task_file", type=str, help="Input task file for decomposition")
    parser.add_argument("--output_name", type=str, help="Output file name for decomposition results")
    parser.add_argument("--mode", type=str, choices=["generate", "decompose"], default="generate",
                       help="Mode: generate text or decompose tasks")
    return parser.parse_args()


def extract_json_array_precise(text):
    """Extract JSON array from text with precise pattern matching"""
    pattern = r'\[\s*(?:\{\s*"step":\s*\d+,\s*"description":\s*"[^"]*"\s*\}(?:\s*,\s*)?)+\s*\]'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        json_str = matches[0]
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            return json_str
    return None


def task_decomposition_vllm(llm, tokenizer, task_content, sampling_params):
    """
    Use local vLLM to perform task decomposition instead of cloud API
    """
    system_prompt = """You are a task decomposition expert. Now you need to reverse-engineer the process of breaking down complex queries into subtasks based on the given information.
## Input Information:
### 1. Available Tool List:
[{"name": Tool Name 1, "description":Function description}, {"name": Tool Name 2, "description": Function description}...]
### 2. Original Query:
[Insert the original query here]
### 3. Final Tool Invocation Results:
[Call Tool A, Call Tool B]
...
## Task Requirements:
Based on the above information, please reverse-engineer a reasonable subtask decomposition process. Just output the subtask list in following format.

## Output Format:
[{"step":1 , "description": Subtask 1}, {"step": 2, "description": Subtask 2}...]

<example_1>
Tool List:[{'name': 'mv', 'description': 'This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Move a file or directory from one location to another. so Note that the provided function is in Python 3 syntax.', 'parameters': {'type': 'dict', 'properties': {'source': {'type': 'string', 'description': 'Source name of the file or directory to move. Source must be local to the current directory.'}, 'destination': {'type': 'string', 'description': 'The destination name to move the file or directory to. Destination must be local to the current directory and cannot be a path. If destination is not an existing directory like when renaming something, destination is the new file name. '}}, 'required': ['source', 'destination']}, 'response': {'type': 'dict', 'properties': {'result': {'type': 'string', 'description': 'The result of the move operation.'}}}},{'name': 'cd', 'description': 'This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Change the current working directory to the specified folder. Note that the provided function is in Python 3 syntax.', 'parameters': {'type': 'dict', 'properties': {'folder': {'type': 'string', 'description': 'The folder of the directory to change to. You can only change one folder at a time. '}}, 'required': ['folder']}, 'response': {'type': 'dict', 'properties': {'current_working_directory': {'type': 'string', 'description': 'The new current working directory path.'}}}}, {'name': 'mkdir', 'description': 'This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Create a new directory in the current directory. Note that the provided function is in Python 3 syntax.', 'parameters': {'type': 'dict', 'properties': {'dir_name': {'type': 'string', 'description': 'The name of the new directory at current directory. You can only create directory at current directory.'}}, 'required': ['dir_name']}, 'response': {'type': 'dict', 'properties': {}}}]
Query: Copy all the text files of the 'Quarter1_Reports' directory and place it in a new directory naming it 'Archived_Quarter1'.\n
Tool Invocation Results: [cd(folder='document'), mkdir(dir_name='temp'), mv(source='final_report.pdf',destination='temp')]

SubTasks:
 [
  {
    "step": 1,
    "description": "Use the 'cd' tool to navigate to the 'document' directory."
  },
  {
    "step": 2,
    "description": "Use the 'mkdir' tool to create a new directory named 'temp' within the 'document' directory if it doesn't already exist."
  },
  {
    "step": 3,
    "description": "Use the 'mv' tool to move the file 'final_report.pdf' from the current directory to the newly created 'temp' directory."
  }
]
</example_1>
Please begin the analysis:"""

    # Format the full prompt
    full_prompt = f"### Instruction: {system_prompt}\n\n### Input: {task_content}\n\n### Response: "

    # Generate using vLLM
    outputs = llm.generate([full_prompt], sampling_params)

    # Extract the generated text
    generated_text = outputs[0].outputs[0].text

    # Try to extract JSON array from the generated text
    split_result = extract_json_array_precise(generated_text)

    if split_result is None:
        # If precise extraction fails, try to find any JSON array
        try:
            json_pattern = r'\[.*\]'
            matches = re.findall(json_pattern, generated_text, re.DOTALL)
            if matches:
                split_result = json.loads(matches[0])
        except:
            print(f"Failed to extract JSON from generated text: {generated_text}")
            return None

    return split_result


def process_task_decomposition(llm, tokenizer, task_file, output_name, sampling_params):
    """
    Process task decomposition for all tasks in the input file
    """
    result = []

    with open(task_file, 'r') as f:
        query_mt_list = json.load(f)

    for i, query_mt in enumerate(query_mt_list):
        try:
            # Extract the required information
            functions = query_mt['conversations'][0]['content'].split('Here is a list of functions in JSON format that you can invoke.\n')[-1]
            query_content = query_mt['conversations'][-1]['content']
            gt_content = query_mt['ground_truth']

            # Create task content for decomposition
            task_content = f"Tool List:{functions}\n Query:{query_content} \n Tool Invocation Results:{gt_content} SubTasks:"

            # Use local vLLM for task decomposition
            split_result = task_decomposition_vllm(llm, tokenizer, task_content, sampling_params)

            if split_result is None:
                print(f'Failed to decompose task {i}')
                continue

            # Add subtasks to the query_mt
            query_mt['sub_tasks'] = split_result
            result.append(query_mt)

            # Save intermediate results
            with open(output_name, 'w') as f:
                json.dump(result, f, indent=True, ensure_ascii=False)

            print(f'Processed task {i+1}/{len(query_mt_list)}')

        except Exception as e:
            print(f'Error processing task {i}: {e}')
            continue

    print(f'Task decomposition completed. Results saved to {output_name}')


def compose_reasoning_process_offline_irrelevance(task_file, composed_reasoning_file):
    """
    Compose reasoning process for irrelevant tasks
    """
    with open(task_file, 'r') as f:
        tasks = json.load(f)

    new_tasks = []
    i = 0

    for task in tasks:
        answer = task['ground_truth']
        first_reply = True

        # Check if this is a first reply
        for conv in task['conversations']:
            if 'tool_response' in conv['content']:
                first_reply = False
                break

        if first_reply:
            query = task['conversations'][-1]['content']
            full_reasoning = f"""OK, let me analyze the overall task: {query}\n However, I find that the user's query is missing key information, or the user's query cannot be fulfilled with the current list of functions.\n"""
            task['conversations'].append({'role':'assistant', 'content': f"<think>\n{full_reasoning}\n</think>\n\n{answer}"})
            task['id'] = i
            i += 1
            del task['ground_truth']
            new_tasks.append(task)

    with open(composed_reasoning_file, 'w') as f:
        json.dump(new_tasks, f, indent=True, ensure_ascii=False)

    print(f'Composed reasoning process saved to {composed_reasoning_file}')


# NOTE: `gpu_queue, data_frac` needs to be at the end (others handled by partial)
def run_process_on_gpu(
    model_path, input_dir, frac_len, world_size, output_dir, split, mode, task_file, output_name, gpu_queue, data_frac
):
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running on GPU: {gpu_id}")

    if mode == "decompose":
        # Task decomposition mode
        generate_on_single_gpu_decompose(
            model_path, task_file, output_name, world_size
        )
    else:
        # Text generation mode
        generate_on_single_gpu(
            model_path, input_dir, frac_len, data_frac, world_size, output_dir, split
        )

    gpu_queue.put(gpu_id)


def generate_on_single_gpu_decompose(model_path, task_file, output_name, world_size):
    """
    Generate task decomposition using local vLLM
    """
    print(f"Running task decomposition on GPU...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        stop=["###", "</s>", "<|im_end|>"]
    )

    # Process task decomposition
    process_task_decomposition(llm, tokenizer, task_file, output_name, sampling_params)


def generate_on_single_gpu(
    model_path, input_dir, frac_len, data_frac, world_size, output_dir, split
):
    # TODO: the generation can be decoupled to use async engine and multiple clients
    # to accelerate, which will amortize the loading time
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating on GPU with data fraction {data_frac}...")
    # load a base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
    )

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)

    # load data
    data = load_dataset(input_dir, split=split)
    data = data.shuffle(seed=42)
    if frac_len > 0:
        sub_len = frac_len
        if sub_len * (data_frac + 1) > len(data):
            data = data[sub_len * data_frac :]["real"]
        else:
            data = data[sub_len * data_frac : sub_len * (data_frac + 1)]["real"]

    prompts_all = [
        "### Instruction: " + data[idx][0]["content"] + "\n\n### Response: "
        for idx in range(len(data))
    ]
    prompts_old = [data[idx][0]["content"] for idx in range(len(data))]
    corrects_all = [data[idx][1]["content"] for idx in range(len(data))]

    start = time.time()

    # run vllm
    results_gathered = list(
        map(lambda x: x.outputs[0].text, llm.generate(prompts_all, sampling_params))
    )

    results = [r.replace("</s>", "").lstrip() for r in results_gathered]

    timediff = time.time() - start
    print(f"time elapsed: {timediff}")

    # collecting data
    for idx in range(len(corrects_all)):
        d = {
            "real": [
                {"role": "user", "content": prompts_old[idx]},
                {"role": "assistant", "content": corrects_all[idx]},
            ],
            "generated": [
                {"role": "user", "content": prompts_old[idx]},
                {"role": "assistant", "content": results[idx]},
            ],
        }
        if split == "test":
            filename = f"{output_dir}/loser_{data_frac}_test.jsonl"
        else:
            filename = f"{output_dir}/loser_{data_frac}.jsonl"
        with open(filename, "a") as f:
            json.dump(d, f)
            f.write("\n")


def main():
    start = time.time()
    mp.set_start_method("spawn", force=True)
    args = parse_arguments()

    if args.mode == "decompose":
        # Task decomposition mode
        if not args.task_file or not args.output_name:
            print("Error: --task_file and --output_name are required for decompose mode")
            return

        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        # Check if the model is already downloaded
        model_path = args.model

        if not model_path.startswith("/"):  # hub path
            filepath = try_to_load_from_cache(model_path, "config.json")
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_directory = cache_dir / f"models--{model_path.replace('/', '--')}"

            print(f"checking cache results: {filepath}")
            if isinstance(filepath, str):
                print(f"Model {model_path} is already downloaded.")
            else:
                print(f"Model {model_path} is not downloaded, will be downloaded now")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                print(f"Model {model_path} downloaded.")
                del tokenizer
                del model
        else:  # local path
            model_directory = model_path
        print(f"model directory: {model_directory}")

        # Run task decomposition on single GPU
        generate_on_single_gpu_decompose(args.model, args.task_file, args.output_name, args.tp_per_worker)

        # Compose reasoning process
        composed_file = args.output_name.replace("sub_task", "train_irrelevance")
        compose_reasoning_process_offline_irrelevance(args.output_name, composed_file)

    else:
        # Text generation mode (original functionality)
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        # Check if the model is already downloaded
        model_path = args.model

        if not model_path.startswith("/"):  # hub path
            filepath = try_to_load_from_cache(model_path, "config.json")
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_directory = cache_dir / f"models--{model_path.replace('/', '--')}"

            print(f"checking cache results: {filepath}")
            if isinstance(filepath, str):
                print(f"Model {model_path} is already downloaded.")
            else:
                print(f"Model {model_path} is not downloaded, will be downloaded now")

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                print(f"Model {model_path} downloaded.")
                del tokenizer
                del model

        else:  # local path
            model_directory = model_path
        print(f"model directory: {model_directory}")

        # Create a pool of processes. Each process will run on a separate GPU.
        with mp.Manager() as manager:
            gpu_queue = manager.Queue()  # Create a Manager Queue
            # Add gpu_id to the queue
            for i in range(num_gpus):
                gpu_queue.put(i)

            with mp.Pool(processes=num_gpus) as pool:
                # Partial function with all arguments except the one that changes per process (data_frac)
                func = partial(
                    run_process_on_gpu,
                    args.model,
                    args.input_dir,
                    args.frac_len,
                    args.tp_per_worker,
                    args.output_dir,
                    args.split,
                    args.mode,
                    args.task_file if args.task_file else None,
                    args.output_name if args.output_name else None,
                )

                # for each data_frac, scheduling one task
                res_futs = []
                for data_frac in range(args.num_data_frac):
                    res_futs.append(
                        pool.apply_async(
                            func,
                            (
                                gpu_queue,
                                data_frac,
                            ),
                        )
                    )

                for res in res_futs:
                    res.get()

    print(f"finished in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()