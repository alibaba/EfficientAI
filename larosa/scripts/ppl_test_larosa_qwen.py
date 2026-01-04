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

import sys,os
# sys.path.append('../')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'utils'))

import torch
from tqdm import tqdm
import os
import argparse
from datasets import load_dataset
import transformers
from utils.eval_ppl import eval_ppl, eval_ppl_wikitext

from inference.modeling_qwen2_larosa import Qwen2ForCausalLM

from utils.data import get_dataset

from transformers import AutoConfig, AutoTokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf",help='Name of the model to use')
    parser.add_argument('--larosa_path', type=str, required=True,help='Path to the larosa input')
    # parser.add_argument('--greedy_flag', action='store_true', help='Flag for greedy')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Sparsity level')
    args = parser.parse_args()

    config = transformers.AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    config.use_cache = False
    config._attn_implementation= "flash_attention_2"
    config.torch_dtype = 'bfloat16'
    config.sparse_level = args.sparsity
    config.Q_path = args.larosa_path

    model = Qwen2ForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map='auto', config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map='cuda', use_fast=True, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    print("="*40)
    with torch.no_grad():
        dense_ppl = eval_ppl_wikitext(model, tokenizer, device="cuda", dataset=dataset, debug=False)
    print(f"Larosa PPL: {dense_ppl}")