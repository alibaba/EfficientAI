#!/bin/bash
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export HF_ENDPOINT="https://hf-mirror.com"


ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes 8 \
    --main_process_port 6002 \
    src/open_r1/sft_moi_qwen3.py \
    --config recipes/Qwen3/sft/Qwen3_8B_d_core_0901_16k_concat.yaml
