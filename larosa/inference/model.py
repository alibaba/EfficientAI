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


from torch import nn

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralConfig,
    MistralForCausalLM
)

from transformers.models.qwen2.modeling_qwen2 import(
    Qwen2DecoderLayer,
    Qwen2Config,
    Qwen2ForCausalLM
)

from transformers.models.llama.configuration_llama import LlamaConfig

from inference.mlp import _monkeypatch_mlp, _monkeypatch_mlp_larosa
from inference.self_attn import _monkeypatch_self_attn, _monkeypatch_self_attn_larosa
from utils.utils import ActivationModule, get_layer_greedy_sparsities


def _monkeypatch_layer(layer, path, grabbing_mode=False):
    layer.path = path
    layer.grabbing_mode = grabbing_mode
    layer.mlp = _monkeypatch_mlp(layer.mlp, f"{path}/mlp", grabbing_mode=grabbing_mode)
    layer.self_attn = _monkeypatch_self_attn(layer.self_attn, f"{path}/self_attn", grabbing_mode=grabbing_mode)
    return layer

class SparseModelMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Extract custom arguments
        histogram_path = kwargs.pop('histogram_path', None)
        grab_acts = kwargs.pop('grab_acts', False)

        greedy_sparsity_path = kwargs.pop('greedy_sparsity_path', None)
        greedy_sparsity_level = kwargs.pop('greedy_sparsity_level', None)

        uniform_sparsity = kwargs.pop('uniform_sparsity', None)
        mlp_sparsity = kwargs.pop('mlp_sparsity', None)
        self_attn_sparsity = kwargs.pop('self_attn_sparsity', None)
        apply_prefill = kwargs.pop('apply_prefill', True)

        # Load the config
        config = kwargs.get('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            kwargs.pop('config', None)

        # Create the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        # Apply sparse layers if histogram_path is provided
        assert histogram_path is not None, "histogram_path must be provided"
        os.makedirs(histogram_path, exist_ok=True)



        model.set_grabbing_mode(grab_acts)
        model.build_sparse_layers(histogram_path, grab_acts)

        if greedy_sparsity_path is not None:
            assert greedy_sparsity_level is not None, "greedy_sparsity_level must be provided"
            model.load_greedy_sparsities(greedy_sparsity_path, greedy_sparsity_level)
        elif uniform_sparsity is not None:
            model.set_uniform_sparsity(uniform_sparsity)
        elif mlp_sparsity is not None or self_attn_sparsity is not None:
            if mlp_sparsity is not None:
                model.set_mlp_sparsity(mlp_sparsity)
            if self_attn_sparsity is not None:
                model.set_self_attn_sparsity(self_attn_sparsity)
        elif not grab_acts:
            model.reset_sparsities()

        if not grab_acts:
            model.set_apply_prefill(apply_prefill)

        return model

    def set_grabbing_mode(self, mode):
        for layer in self.model.layers:
            layer.mlp.grabbing_mode = mode
            layer.self_attn.grabbing_mode = mode

    def set_apply_prefill(self, apply_prefill):
        for layer in self.model.layers:
            for proj in ['q', 'k', 'v', 'o']:
                layer.self_attn.sparse_fns[proj].apply_prefill = apply_prefill
            for proj in ['gate', 'up', 'down']:
                layer.mlp.sparse_fns[proj].apply_prefill = apply_prefill


    def build_sparse_layers(self, histogram_path, grab_acts):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        layers = []
        os.makedirs(histogram_path, exist_ok=True)

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, LlamaDecoderLayer) or isinstance(layer, MistralDecoderLayer) or isinstance(layer, Qwen2DecoderLayer):
                layers.append(_monkeypatch_layer(layer, path=f"{histogram_path}/layer-{i}", grabbing_mode=grab_acts))
            else:
                raise ValueError(f"Unknown layer type: {type(layer)}")
        
        self.model.layers = nn.ModuleList(layers)

    
    def load_greedy_sparsities(self, greedy_sparsity_path, greedy_sparsity_level):
        layer_sparsity_levels = [greedy_sparsity_level] * len(self.model.layers)
        sparsities = get_layer_greedy_sparsities(layer_sparsity_levels, greedy_sparsity_path)
        self.set_sparsities(sparsities)
    
    def reset_sparsities(self):
        self.set_uniform_sparsity(0)

    def set_mlp_sparsity(self, sparsity):
        for layer in self.model.layers:
            layer.mlp.sparse_fns['gate'].set_threshold(sparsity)
            layer.mlp.sparse_fns['up'].set_threshold(sparsity)
            layer.mlp.sparse_fns['down'].set_threshold(sparsity)

    def set_self_attn_sparsity(self, sparsity):
        for layer in self.model.layers:
            layer.self_attn.sparse_fns['q'].set_threshold(sparsity)
            layer.self_attn.sparse_fns['k'].set_threshold(sparsity)
            layer.self_attn.sparse_fns['v'].set_threshold(sparsity)
            layer.self_attn.sparse_fns['o'].set_threshold(sparsity)
    
    def set_uniform_sparsity(self, sparsity):
        self.set_mlp_sparsity(sparsity)
        self.set_self_attn_sparsity(sparsity)

    def set_sparsities(self, sparsities):
        for proj, sparses in sparsities.items():
            if proj in ['q', 'k', 'v', 'o']:
                for layer, sparsity in zip(self.model.layers, sparses):
                    layer.self_attn.sparse_fns[proj].set_threshold(sparsity)
            elif proj in ['gate', 'up', 'down']:
                for layer, sparsity in zip(self.model.layers, sparses):
                    layer.mlp.sparse_fns[proj].set_threshold(sparsity)

def _monkeypatch_layer_larosa(layer, path, grabbing_mode=False, rot=False):
    layer.path = path
    layer.grabbing_mode = grabbing_mode
    layer.rot = rot
    layer.mlp = _monkeypatch_mlp_larosa(layer.mlp, f"{path}/mlp", grabbing_mode=grabbing_mode, rot=rot)
    layer.self_attn = _monkeypatch_self_attn_larosa(layer.self_attn, f"{path}/self_attn", grabbing_mode=grabbing_mode, rot=rot)
    return layer

class SparseModelMixinRotate:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Extract custom arguments
        histogram_path = kwargs.pop('histogram_path', None)
        grab_acts = kwargs.pop('grab_acts', False)
        rot = kwargs.pop('rot', False)

        greedy_sparsity_path = kwargs.pop('greedy_sparsity_path', None)
        greedy_sparsity_level = kwargs.pop('greedy_sparsity_level', None)

        uniform_sparsity = kwargs.pop('uniform_sparsity', None)
        mlp_sparsity = kwargs.pop('mlp_sparsity', None)
        self_attn_sparsity = kwargs.pop('self_attn_sparsity', None)
        apply_prefill = kwargs.pop('apply_prefill', True)

        # Load the config
        config = kwargs.get('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            kwargs.pop('config', None)

        # Create the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        # Apply sparse layers if histogram_path is provided
        assert histogram_path is not None, "histogram_path must be provided"
        os.makedirs(histogram_path, exist_ok=True)


        model.set_grabbing_mode(grab_acts)
        model.build_sparse_layers_larosa(histogram_path, grab_acts, rot)

        if greedy_sparsity_path is not None:
            assert greedy_sparsity_level is not None, "greedy_sparsity_level must be provided"
            model.load_greedy_sparsities(greedy_sparsity_path, greedy_sparsity_level)
        elif uniform_sparsity is not None:
            model.set_uniform_sparsity(uniform_sparsity)
        elif mlp_sparsity is not None or self_attn_sparsity is not None:
            if mlp_sparsity is not None:
                model.set_mlp_sparsity(mlp_sparsity)
            if self_attn_sparsity is not None:
                model.set_self_attn_sparsity(self_attn_sparsity)
        elif not grab_acts:
            model.reset_sparsities()

        if not grab_acts:
            model.set_apply_prefill(apply_prefill)

        return model

    def set_grabbing_mode(self, mode):
        for layer in self.model.layers:
            layer.mlp.grabbing_mode = mode
            layer.self_attn.grabbing_mode = mode

    def set_apply_prefill(self, apply_prefill):
        for layer in self.model.layers:
            for proj in ['q', 'k', 'v', 'o']:
                layer.self_attn.sparse_fns[proj].apply_prefill = apply_prefill
            for proj in ['gate', 'up', 'down']:
                layer.mlp.sparse_fns[proj].apply_prefill = apply_prefill


    def build_sparse_layers_larosa(self, histogram_path, grab_acts, rot):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        layers = []
        os.makedirs(histogram_path, exist_ok=True)

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, LlamaDecoderLayer) or isinstance(layer, MistralDecoderLayer) or isinstance(layer, Qwen2DecoderLayer):
                layers.append(_monkeypatch_layer_larosa(layer, path=f"{histogram_path}/layer-{i}", grabbing_mode=grab_acts, rot=rot))
            else:
                raise ValueError(f"Unknown layer type: {type(layer)}")
        
        self.model.layers = nn.ModuleList(layers)

    
    def load_greedy_sparsities(self, greedy_sparsity_path, greedy_sparsity_level):
        layer_sparsity_levels = [greedy_sparsity_level] * len(self.model.layers)
        sparsities = get_layer_greedy_sparsities(layer_sparsity_levels, greedy_sparsity_path)
        self.set_sparsities(sparsities)
    
    def reset_sparsities(self):
        self.set_uniform_sparsity(0)

    def set_mlp_sparsity(self, sparsity):
        for layer in self.model.layers:
            layer.mlp.sparse_fns['gate'].set_threshold(sparsity)
            layer.mlp.sparse_fns['up'].set_threshold(sparsity)
            layer.mlp.sparse_fns['down'].set_threshold(sparsity)

    def set_self_attn_sparsity(self, sparsity):
        for layer in self.model.layers:
            layer.self_attn.sparse_fns['q'].set_threshold(sparsity)
            layer.self_attn.sparse_fns['k'].set_threshold(sparsity)
            layer.self_attn.sparse_fns['v'].set_threshold(sparsity)
            layer.self_attn.sparse_fns['o'].set_threshold(sparsity)
    
    def set_uniform_sparsity(self, sparsity):
        self.set_mlp_sparsity(sparsity)
        self.set_self_attn_sparsity(sparsity)

    def set_sparsities(self, sparsities):
        for proj, sparses in sparsities.items():
            if proj in ['q', 'k', 'v', 'o']:
                for layer, sparsity in zip(self.model.layers, sparses):
                    layer.self_attn.sparse_fns[proj].set_threshold(sparsity)
            elif proj in ['gate', 'up', 'down']:
                for layer, sparsity in zip(self.model.layers, sparses):
                    layer.mlp.sparse_fns[proj].set_threshold(sparsity)

class LlamaSparseConfig(LlamaConfig):
    model_type = "llama_sparse"

class MistralSparseConfig(MistralConfig):
    model_type = "mistral_sparse"

class Qwen2SparseConfig(Qwen2Config):
    model_type = "qwen2_sparse"

class Qwen2SparseForCausalLM(SparseModelMixin, Qwen2ForCausalLM):
    config_class = Qwen2SparseConfig
    _no_split_modules = ["Qwen2DecoderLayer"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

class LlamaSparseForCausalLM(SparseModelMixin, LlamaForCausalLM):
    # config_class = LlamaSparseConfig
    config_class = LlamaSparseConfig
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

class LlamaSparseForCausalLMRotate(SparseModelMixinRotate, LlamaForCausalLM):
    # config_class = LlamaSparseConfig
    config_class = LlamaSparseConfig
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

class Qwen2SparseForCausalLMRotate(SparseModelMixinRotate, Qwen2ForCausalLM):
    # config_class = LlamaSparseConfig
    config_class = Qwen2SparseConfig
    _no_split_modules = ["Qwen2DecoderLayer"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

class MistralSparseForCausalLMRotate(SparseModelMixinRotate, MistralForCausalLM):
    # config_class = LlamaSparseConfig
    config_class = MistralSparseConfig
    _no_split_modules = ["MistralDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()
class MistralSparseForCausalLM(SparseModelMixin, MistralForCausalLM):
    config_class = MistralSparseConfig
    _no_split_modules = ["MistralDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()