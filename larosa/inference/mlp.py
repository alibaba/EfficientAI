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

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

import types
from torch import nn
import torch
import pickle

from utils.utils import ActivationModule, Distribution, SparsifyFn, get_module_device

def exchange_row_col(_tensor, i, j):
    tensor = _tensor.detach().clone()
    assert isinstance(tensor, torch.Tensor)
    indices_row = torch.arange(tensor.size(0))
    indices_row[i], indices_row[j] = indices_row[j].item(), indices_row[i].item()
    tensor = tensor[indices_row]

    indices_col = torch.arange(tensor.size(1))
    indices_col[i], indices_col[j] = indices_col[j].item(), indices_col[i].item()
    tensor = tensor[:, indices_col]
    return tensor


def _monkeypatch_mlp(mlp, file_path, grabbing_mode=False):
    mlp.forward_old = mlp.forward
    mlp.forward = types.MethodType(_mlp_forward, mlp)

    mlp.file_path = file_path
    mlp.grabbing_mode = grabbing_mode

    if not grabbing_mode:
        mlp.distrs = {}
        mlp.distrs['h1'] = Distribution(file_path, hidden_type='h1')
        mlp.distrs['h2'] = Distribution(file_path, hidden_type='h2')


        mlp.sparse_fns = nn.ModuleDict({
            'gate': SparsifyFn(mlp.distrs['h1']).to(get_module_device(mlp)),
            'up': SparsifyFn(mlp.distrs['h1']).to(get_module_device(mlp)),
            'down': SparsifyFn(mlp.distrs['h2']).to(get_module_device(mlp)),
        })

    mlp.activation_module = ActivationModule(file_path)

    return mlp

def count_zero(tensor1, tensor2, tensor3):
    zero_count = (tensor1 == 0.0).sum().item()
    zero_count += (tensor2 == 0.0).sum().item()
    zero_count += (tensor3 == 0.0).sum().item()
    total_count = tensor1.numel() + tensor2.numel() + tensor3.numel()
    sparsity = zero_count / total_count
    return sparsity

def count_zero_solo(tensor1):
    zero_count = (tensor1 == 0.0).sum().item()
    total_count = tensor1.numel() 
    sparsity = zero_count / total_count
    return sparsity

def _mlp_forward(self, x, activation_module=None):
    if hasattr(self, 'config') and self.config.pretraining_tp > 1:
        # TODO: UNTESTED

        assert 1 == 0, "Pretraining TP > 1 not implemented yet"
    else:
        if self.grabbing_mode:
            self.activation_module.grab_activations(x, 'h1')

            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            self.activation_module.grab_activations(intermediate_states, 'h2')
            down_proj = self.down_proj(intermediate_states)
        else:
            x_gate = self.sparse_fns['gate'](x)
            x_up = self.sparse_fns['up'](x)

            intermediate_states = self.act_fn(self.gate_proj(x_gate)) * self.up_proj(x_up)
            intermediate_states = self.sparse_fns['down'](intermediate_states)

            self.infer_sparsity_h1 = count_zero_solo(x_gate)
            self.infer_sparsity_h2 = count_zero_solo(intermediate_states)

            down_proj = self.down_proj(intermediate_states)

    return down_proj

def _monkeypatch_mlp_larosa(mlp, file_path, grabbing_mode=False, rot=False):
    mlp.forward_old = mlp.forward
    mlp.forward = types.MethodType(_mlp_forward_larosa, mlp)

    mlp.file_path = file_path
    mlp.grabbing_mode = grabbing_mode
    mlp.rot = rot

    mlp.activation_module = ActivationModule(file_path)

    if not grabbing_mode:
        mlp.distrs = {}
        mlp.distrs['h1'] = Distribution(file_path, hidden_type='h1')
        mlp.distrs['h2'] = Distribution(file_path, hidden_type='h2')


        mlp.sparse_fns = nn.ModuleDict({
            'gate': SparsifyFn(mlp.distrs['h1']).to(get_module_device(mlp)),
            'up': SparsifyFn(mlp.distrs['h1']).to(get_module_device(mlp)),
            'down': SparsifyFn(mlp.distrs['h2']).to(get_module_device(mlp)),
        })

    if rot:
        mlp.D, mlp.inv_D = mlp.activation_module.load_D_and_inv_D()

    return mlp


def _mlp_forward_larosa(self, x, activation_module=None):
    if hasattr(self, 'config') and self.config.pretraining_tp > 1:
        # TODO: UNTESTED
        assert 1 == 0, "Pretraining TP > 1 not implemented yet"
    else:
        if self.grabbing_mode:
            if self.rot: 
                x_float = x.double()
                rot_x = torch.matmul(x_float, self.D)

                recover_x = torch.matmul(rot_x, self.inv_D).to(x.dtype)

                if torch.allclose(x, recover_x, atol=1e-2):
                    print("The reconstructed activations are approximately equal to the original activations.")
                else:
                    print("***The reconstructed matrix have significant deviation from the original matrix***")
                self.activation_module.grab_activations(rot_x.to(x.dtype), 'h1')
                intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                self.activation_module.grab_activations(intermediate_states, 'h2')
                down_proj = self.down_proj(intermediate_states)
            else:
                self.activation_module.grab_activations(x, 'h1')
            # x = x.half()
                intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                self.activation_module.grab_activations(intermediate_states, 'h2')
                down_proj = self.down_proj(intermediate_states)
        else:
            if self.rot:  
                x_double = x.double()
                D = self.D.to(x.device)
                gaussian_x = torch.matmul(x_double, D)
                x_gate = self.sparse_fns['gate'](gaussian_x)
                x_up = self.sparse_fns['up'](gaussian_x)
                inv_D = self.inv_D.to(x.device)
                tmp_gate = x_gate
                x_gate = torch.matmul(tmp_gate, inv_D).to(x.dtype)
                tmp_up = x_up
                x_up = torch.matmul(tmp_up, inv_D).to(x.dtype)
            else:
                x_gate = self.sparse_fns['gate'](x)
                x_up = self.sparse_fns['up'](x)


            intermediate_states = self.act_fn(self.gate_proj(x_gate)) * self.up_proj(x_up)
            intermediate_states = self.sparse_fns['down'](intermediate_states)

            self.infer_sparsity_h1 = count_zero_solo(tmp_gate)
            self.infer_sparsity_h2 = count_zero_solo(intermediate_states)

            down_proj = self.down_proj(intermediate_states)

    return down_proj