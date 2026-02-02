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

from typing import Any, Callable, Optional, Type, Union, Dict


import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DefaultDataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from .modeling_qwen2_moi import Qwen2ForCausalLM

class MOI_SFTTrainer(SFTTrainer):
    def __init__(
        self,
        data_collator = DefaultDataCollator(),  # type: ignore
        sft_attention_type: str = "packing_mask",
        first_n_token_import: int = 0,
        first_n_token_weight: float = 5.0,
        **kwargs
    ):
        self.sft_attention_type = sft_attention_type
        self.first_n_token_import = int(first_n_token_import)
        self.first_n_token_weight = float(first_n_token_weight)
        super().__init__(data_collator=data_collator, **kwargs)

    def _create_model_from_path(self, model_path: str, args: SFTConfig) -> PreTrainedModel:
        """Creates a model from a path or model identifier. moi add flash_attention_mask in QwenModel, we should load model locally"""
        model_init_kwargs = args.model_init_kwargs or {}
        # Handle torch dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        if args.gradient_checkpointing:
            model_init_kwargs["use_cache"] = False

        # Create model
        # model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        model = Qwen2ForCausalLM.from_pretrained(model_path, **model_init_kwargs)
        return model

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str
    ) -> Union[Dataset, IterableDataset]:
        sft_attention_type = self.sft_attention_type
        reuqired_columns = ["input_ids", "labels", "attention_mask", "position_ids", "flash_attention_mask", "weights"]
        remove_columns = list(set(dataset.column_names) - set(reuqired_columns))
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        if sft_attention_type == "packing_mask":
            dataset = dataset.map(
                self.map_prepare_packing_mask_dataset, 
                batched=True, 
                fn_kwargs={"processing_class": processing_class, "max_length": args.max_length, "first_n_token_import": self.first_n_token_import, "first_n_token_weight": self.first_n_token_weight}, 
                remove_columns=remove_columns,
                **map_kwargs
            )
            return dataset
        elif sft_attention_type == "only_last_assistant":
            dataset = dataset.map(
                self.map_prepare_packing_mask_dataset_only_last_assistant, 
                batched=True, 
                fn_kwargs={"processing_class": processing_class, "max_length": args.max_length, "first_n_token_import": self.first_n_token_import, "first_n_token_weight": self.first_n_token_weight}, 
                remove_columns=remove_columns,
                **map_kwargs
            )
            return dataset
        else:
            assert False, "the processing type should be in ['packing_mask']"

    def map_prepare_packing_mask_dataset(
        self,
        examples: dict[str, list],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        max_length: int,
        ignor_token_id: int = -100,
        first_n_token_import = 0,
        first_n_token_weight = 5.0
    ) -> Union[Dataset, IterableDataset]:
        roles = {"system": "<|im_start|>system", "user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

        im_start = processing_class('<|im_start|>').input_ids
        im_end = processing_class('<|im_end|>').input_ids
        nl_tokens = processing_class('\n').input_ids
        _system = processing_class('system').input_ids + nl_tokens
        _user = processing_class('user').input_ids + nl_tokens
        _assistant = processing_class('assistant').input_ids + nl_tokens

        # Apply prompt templates
        input_ids, targets= [], []
        flash_attention_masks = []
        last_mask_indice = 0
        position_ids = []
        weights = []
        for conversations in examples['conversations']:
            if conversations[0]['role'] == 'system':
                system_message = conversations[0]["content"]
                conversations = conversations[1:]

            input_id, target = [], []
            flash_attention_mask = []
            weight = []
            system = im_start + _system + processing_class(system_message).input_ids + im_end + nl_tokens
            input_id += system
            target += im_start + [ignor_token_id] * (len(system) - 3) + im_end + nl_tokens
            weight += [1.0] + [0] * (len(system) - 3) + [1.0] + [1.0]
            last_mask_indice = 1
            flash_attention_mask += [last_mask_indice] * len(target)

            position_id = []
            start_pos_id = 0
            for pos in range(start_pos_id, start_pos_id + len(input_id)):
                position_id.append(pos)
            start_pos_id += len(input_id) 

            assert len(input_id) == len(target)
            for j, sentence in enumerate(conversations):
                if sentence['role'] in roles.keys():
                    role = roles[sentence["role"]]
                    weight_cur = sentence['weight'] if 'weight' in sentence else 1.0
                else:
                    role = "<|im_start|>" + sentence['role']
                _input_id = processing_class(role).input_ids + nl_tokens + \
                    processing_class(sentence["content"]).input_ids + im_end + nl_tokens
                input_id += _input_id
                if role == '<|im_start|>user':
                    _target = im_start + [ignor_token_id] * (len(_input_id)-3) + im_end + nl_tokens
                    _weight = [1.0] + [0.0] * (len(_input_id)-3) + [1.0] + [1.0]
                    for pos in range(start_pos_id, start_pos_id + len(_input_id)):
                        position_id.append(pos)
                    start_pos_id += len(_input_id) 
                elif role == '<|im_start|>system':
                    _target = im_start + [ignor_token_id] * (len(_input_id)-3) + im_end + nl_tokens
                    _weight = [1.0] + [0.0] * (len(_input_id)-3) + [1.0] + [1.0]
                    last_mask_indice = len(target)
                    start_pos_id = 0 
                    for pos in range(start_pos_id, start_pos_id + len(_input_id)):
                        position_id.append(pos)
                    start_pos_id += len(_input_id) 
                elif role == '<|im_start|>assistant':
                    _target = im_start + [ignor_token_id] * len(processing_class(role).input_ids) + \
                        _input_id[len(processing_class(role).input_ids)+1:-2] + im_end + nl_tokens
                    
                    # assistant返回的前n个token的权重加大，
                    if first_n_token_import > 0:
                        _weight = [1.0] + [0.0] * len(processing_class(role).input_ids) + [weight_cur * first_n_token_weight] * first_n_token_import + [weight_cur] * (len(_input_id)-5-first_n_token_import) + [1.0] + [1.0]
                    else:
                        _weight = [1.0] + [0.0] * len(processing_class(role).input_ids) + [weight_cur] * (len(_input_id)-5) + [1.0] + [1.0]
                    for pos in range(start_pos_id, start_pos_id + len(_input_id)):
                        position_id.append(pos)
                    start_pos_id += len(_input_id) 
                else:
                    _target = im_start + [ignor_token_id] * len(processing_class(role).input_ids) + \
                        _input_id[len(processing_class(role).input_ids)+1:-2] + im_end + nl_tokens
                    _weight = [1.0] + [0.0] * len(processing_class(role).input_ids) + [weight_cur] * (len(_input_id)-3-len(processing_class(role).input_ids)) + [1.0] + [1.0]
                    for pos in range(start_pos_id, start_pos_id + len(_input_id)):
                        position_id.append(pos)
                    start_pos_id += len(_input_id) 
                flash_attention_mask += [last_mask_indice] * len(_target)
                target += _target
                weight += _weight
            
            assert len(input_id) == len(target)
            assert len(input_id) == len(weight)
            assert len(input_id) == len(flash_attention_mask)
            position_id += [k for k in range(len(input_id), max_length)]
            input_id += [processing_class.pad_token_id] * (max_length - len(input_id))
            target += [ignor_token_id] * (max_length - len(target))
            weight += [0.0] * (max_length - len(weight))
            flash_attention_mask += [0] * (max_length - len(flash_attention_mask))
        
            input_ids.append(input_id[:max_length])
            targets.append(target[:max_length])
            weights.append(weight[:max_length])
            flash_attention_masks.append(flash_attention_mask[:max_length])
            position_ids.append(position_id[:max_length])
            
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        targets = torch.tensor(targets, dtype=torch.long)
        weights = torch.tensor(weights, dtype=torch.float32)
        flash_attention_masks = torch.tensor(flash_attention_masks, dtype=torch.int)
        position_ids = torch.tensor(position_ids, dtype=torch.long)

        batch_size, seq_length = input_ids.shape
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        return dict(
            input_ids=input_ids,
            labels=targets,
            weights=weights,
            attention_mask=input_ids.ne(processing_class.pad_token_id),
            position_ids = position_ids,
            flash_attention_mask=flash_attention_masks,
        )

    def map_prepare_packing_mask_dataset_only_last_assistant(
        self,
        examples: dict[str, list],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        max_length: int,
        ignor_token_id: int = -100,
        first_n_token_import = 0,
        first_n_token_weight = 5.0
    ) -> Union[Dataset, IterableDataset]:
        roles = {"system": "<|im_start|>system", "user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

        im_start = processing_class('<|im_start|>').input_ids
        im_end = processing_class('<|im_end|>').input_ids
        nl_tokens = processing_class('\n').input_ids

        # Apply prompt templates
        input_ids, targets= [], []
        flash_attention_masks = []
        last_mask_indice = 0
        position_ids = []
        weights = []
        if not 'single_conv_last_idx' in examples:
            examples['single_conv_last_idx'] = []
            for conversations in examples['conversations']:
                examples['single_conv_last_idx'].append([len(conversations)-1])
        for conversations, single_conv_last_idx in zip(examples['conversations'], examples['single_conv_last_idx']):
            input_id, target = [], []
            flash_attention_mask = []
            weight = []

            position_id = []
            start_pos_id = 0

            for j, sentence in enumerate(conversations):
                if sentence['role'] in roles.keys():
                    role = roles[sentence["role"]]
                    weight_cur = sentence['weight'] if 'weight' in sentence else 1.0
                else:
                    role = "<|im_start|>" + sentence['role']
                _input_id = processing_class(role).input_ids + nl_tokens + \
                    processing_class(sentence["content"]).input_ids + im_end + nl_tokens
                input_id += _input_id
                if role == '<|im_start|>user':
                    _target = im_start + [ignor_token_id] * (len(_input_id)-3) + im_end + nl_tokens
                    _weight = [1.0] + [0.0] * (len(_input_id)-3) + [1.0] + [1.0]
                    for pos in range(start_pos_id, start_pos_id + len(_input_id)):
                        position_id.append(pos)
                    start_pos_id += len(_input_id) 
                elif role == '<|im_start|>system':
                    _target = im_start + [ignor_token_id] * (len(_input_id)-3) + im_end + nl_tokens
                    _weight = [1.0] + [0.0] * (len(_input_id)-3) + [1.0] + [1.0]
                    if j == 0:
                        last_mask_indice = 1
                    else:
                        last_mask_indice = len(target)
                    start_pos_id = 0 
                    for pos in range(start_pos_id, start_pos_id + len(_input_id)):
                        position_id.append(pos)
                    start_pos_id += len(_input_id) 
                elif role == '<|im_start|>assistant':
                    if j in single_conv_last_idx:
                        _target = im_start + [ignor_token_id] * len(processing_class(role).input_ids) + \
                            _input_id[len(processing_class(role).input_ids)+1:-2] + im_end + nl_tokens
                    else:
                        _target = im_start + [ignor_token_id] * (len(_input_id)-3) + im_end + nl_tokens
                    
                    # assistant返回的前n个token的权重加大，
                    if first_n_token_import > 0:
                        _weight = [1.0] + [0.0] * len(processing_class(role).input_ids) + [weight_cur * first_n_token_weight] * first_n_token_import + [weight_cur] * (len(_input_id)-5-first_n_token_import) + [1.0] + [1.0]
                    else:
                        _weight = [1.0] + [0.0] * len(processing_class(role).input_ids) + [weight_cur] * (len(_input_id)-5) + [1.0] + [1.0]
                    for pos in range(start_pos_id, start_pos_id + len(_input_id)):
                        position_id.append(pos)
                    start_pos_id += len(_input_id) 
                else:
                    _target = im_start + [ignor_token_id] * len(processing_class(role).input_ids) + \
                        _input_id[len(processing_class(role).input_ids)+1:-2] + im_end + nl_tokens
                    _weight = [1.0] + [0.0] * len(processing_class(role).input_ids) + [weight_cur] * (len(_input_id)-3-len(processing_class(role).input_ids)) + [1.0] + [1.0]
                    for pos in range(start_pos_id, start_pos_id + len(_input_id)):
                        position_id.append(pos)
                    start_pos_id += len(_input_id) 
                flash_attention_mask += [last_mask_indice] * len(_target)
                target += _target
                weight += _weight
            
            assert len(input_id) == len(target)
            assert len(input_id) == len(weight)
            assert len(input_id) == len(flash_attention_mask)
            position_id += [k for k in range(len(input_id), max_length)]
            input_id += [processing_class.pad_token_id] * (max_length - len(input_id))
            target += [ignor_token_id] * (max_length - len(target))
            weight += [0.0] * (max_length - len(weight))
            flash_attention_mask += [0] * (max_length - len(flash_attention_mask))
        
            input_ids.append(input_id[:max_length])
            targets.append(target[:max_length])
            weights.append(weight[:max_length])
            flash_attention_masks.append(flash_attention_mask[:max_length])
            position_ids.append(position_id[:max_length])
            
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        targets = torch.tensor(targets, dtype=torch.long)
        weights = torch.tensor(weights, dtype=torch.float32)
        flash_attention_masks = torch.tensor(flash_attention_masks, dtype=torch.int)
        position_ids = torch.tensor(position_ids, dtype=torch.long)

        batch_size, seq_length = input_ids.shape
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        return dict(
            input_ids=input_ids,
            labels=targets,
            weights=weights,
            attention_mask=input_ids.ne(processing_class.pad_token_id),
            position_ids = position_ids,
            flash_attention_mask=flash_attention_masks,
        )
    