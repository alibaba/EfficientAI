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

import argparse
import os

import torch
import sys
from datasets import load_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))


from utils.utils import get_tokenizer

from inference.model import LlamaSparseForCausalLMRotate, LlamaSparseConfig
from transformers import AutoConfig, AutoModelForCausalLM
from utils.data import get_dataset
from tqdm import tqdm
import gc

def gaussian_matrix_svd(x):
    X_batch = x.float()
    H = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=x.device)
    H[diag, diag] = H[diag, diag] + damp

    print('H:', H.shape)
    X_eig = torch.linalg.eigh(H)

    D = X_eig[1]
    inv_D = D.T

    mean_value = torch.mean(x)
    std_value = torch.std(x)

    new_mean_value = torch.mean(H)
    new_std_value = torch.std(H)

    print(mean_value, new_mean_value)
    print(std_value, new_std_value)
    print('fp32:', torch.matmul(D, inv_D))

    return D, inv_D

def perform_eigen_decomp(Cov_matrix, per_head=False, num_heads=0):
    # performs eigen decomposition and returns
    # the sorted eigen values and eigen vectors
    if per_head:
        assert num_heads != 0  # cannot use per head and not pass num_heads
        eval = []
        evec = []
        for hd in range(num_heads):
            H = Cov_matrix[hd]
            damp = 0.01 * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[-1]).to(device=H.device)
            H[diag, diag] = H[diag, diag] + damp
            X = torch.linalg.eigh(H.to(torch.float64))
            index = torch.argsort(X[0])
            eval.append(X[0][index])
            evec.append(X[1][:, index])
        eval = torch.stack(eval)
        evec = torch.stack(evec)
    else:
        H = Cov_matrix
        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[-1]).to(device=H.device)
        H[diag, diag] = H[diag, diag] + damp

        X = torch.linalg.eigh(H.to(torch.float64))
        eval = X[0]
        evec = X[1]

    return eval, evec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf",help='Name of the model to use')
    parser.add_argument('--output_path', type=str, required=True,help='Path to the output') # contains 1. model itself, 2. histograms, 3. activations
    args = parser.parse_args()

    AutoConfig.register("llama_sparse", LlamaSparseConfig)

    AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLMRotate)
    tokenizer = get_tokenizer(args.model_name)
    model = LlamaSparseForCausalLMRotate.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2", histogram_path=os.path.join(args.output_path, "histograms"), grab_acts=True, rot=False)

    dataset_wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', streaming=True )['train']
    dataset_wiki = dataset_wiki.skip(0).take(500)

    dataset_alpaca = get_dataset(
        "tatsu-lab/alpaca",
        subset=None,
        split="train",
        size=300
    )

    text = ""
    for sample in tqdm(dataset_wiki):
        text += sample["text"] + "\n\n"

    bsz, seq_len = 10, 2048

    encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=seq_len, return_overflowing_tokens=True, padding="max_length")

    input_ids = encodings.input_ids[:bsz,:].to(device="cuda:0")

    hidden_states = model.model.embed_tokens(input_ids)

    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
    past_key_value=None
    output_attentions = False
    use_cache = False
    cache_position=None
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    hidden_dim = model.model.layers[0].mlp.down_proj.weight.shape[0]
    H_attn = torch.zeros((len(model.model.layers), hidden_dim, hidden_dim), device=model.device)
    H_mlp = torch.zeros((len(model.model.layers), hidden_dim, hidden_dim), device=model.device)

    for i in tqdm(range(len(model.model.layers))):
        layer = model.model.layers[i]
        hidden_states = hidden_states.to(layer.self_attn.q_proj.weight.data.device) 
        hidden_states = layer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]

        activation_mlp = layer.mlp.activation_module.combine_activations()['h1']
        activation_attn = layer.self_attn.activation_module.combine_activations()['h1']

        activation_mlp = activation_mlp.to(model.device)
        activation_attn = activation_attn.to(model.device)

        H_attn[i] += torch.sum(activation_attn.double().mT @ activation_attn.double(), dim=0)  # 
        H_mlp[i] += torch.sum(activation_mlp.double().mT @ activation_mlp.double(), dim=0)  # 

        eval_attn, evec_attn = perform_eigen_decomp(H_attn[i])
        D = evec_attn
        inv_D = D.T
        layer.self_attn.activation_module.save_D_and_inv_D(D, inv_D)

        eval_mlp, evec_mlp = perform_eigen_decomp(H_mlp[i])
        D = evec_mlp
        inv_D = D.T
        layer.mlp.activation_module.save_D_and_inv_D(D, inv_D)

        del layer.mlp.activation_module.activations
        del layer.self_attn.activation_module.activations
        model.model.layers[i] = None

        gc.collect()
        torch.cuda.empty_cache()


    model_rot = LlamaSparseForCausalLMRotate.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2", histogram_path=os.path.join(args.output_path, "histograms"), grab_acts=True, rot=True)
    text = ""
    for sample in tqdm(dataset_alpaca):
        text += sample["text"] + "\n\n"

    bsz, seq_len = 10, 2048

    encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=seq_len, return_overflowing_tokens=True, padding="max_length")

    input_ids = encodings.input_ids[:bsz,:].to(device="cuda:0")
    hidden_states = model_rot.model.embed_tokens(input_ids)

    act_path = os.path.join(args.output_path, "activations")
    os.makedirs(act_path, exist_ok=True)

    for i in tqdm(range(len(model_rot.model.layers))):
        layer = model_rot.model.layers[i]
        hidden_states = hidden_states.to(layer.self_attn.q_proj.weight.data.device) 
        hidden_states = layer(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]

        layer.mlp.activation_module.find_histogram()
        layer.self_attn.activation_module.find_histogram()
        layer.mlp.activation_module.save_histogram()
        layer.self_attn.activation_module.save_histogram()

        del layer.mlp.activation_module.activations
        del layer.self_attn.activation_module.activations
        
        model_rot.model.layers[i] = None

        gc.collect()
        torch.cuda.empty_cache()