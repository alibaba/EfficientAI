#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export RAY_DEBUG=0  # ray 分布式 debug

project_name='D-CORE'
export EXPERIMENT_NAME="DA-GRPO-Qwen3-14B-D-Core-TEMP-16k-MOI-alpha0.1-beta0.5"

rollout_data_dir="rollout_log/da_grpo_Qwen3_14B_alpha_0.1_beta_0.5"
validation_data_dir="validation_log/da_grpo_Qwen3_14B_alpha_0.1_beta_0.5"
set -x

# echo 等待启动...
# sleep 41400


# 训练数据
d_core_train_path=examples/data_preprocess/data/d_core/training_temp_data.parquet
d_core_val_path=examples/data_preprocess/data/d_core/training_temp_data.parquet

train_files="['$d_core_train_path']"
test_files="['$d_core_val_path']"

MODEL_PATH="Qwen/Qwen3-14B"
CKPTS_DIR=${CKPTS_DIR:-"checkpoints/${project_name}/${EXPERIMENT_NAME}"}

# toolrl的超参
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

# 算法配置
adv_estimator=da_grpo
entropy_alpha=0.1
entropy_delta=0.5

policy_loss_mode='vanilla'
loss_agg_mode='token-mean'
adv_clip_ration_low=0.2
adv_clip_ratio_high=0.2
adv_clip_ratio=0.2

use_kl_in_reward=False # grpo的kl不是在reward当中，而是在actor当中使用kl损失
kl_coef=0.0
use_kl_loss=True   # grpo 在actor当中使用kl损失
kl_loss_coef=0.001  # kl损失系数，默认为0.001
kl_loss_type='low_var_kl'

max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 4))
filter_overlong_prompts=True

enable_overlong_buffer=False        # dapo overlong penalty
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=0.1

train_prompt_bsz=32
n_resp_per_prompt=8
train_prompt_mini_bsz=8

# Ray
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# actor rollout 采样参数
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=4
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True    # param offload
gen_tp=4    # vllm tensor parallel size
fsdp_size=8 # fsdp size

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.entropy_alpha=${entropy_alpha} \
    algorithm.entropy_delta=${entropy_delta} \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.clip_ratio_low=${adv_clip_ration_low} \
    actor_rollout_ref.actor.clip_ratio_high=${adv_clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio=${adv_clip_ratio} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=3 \
    trainer.val_before_train=True \
    trainer.log_val_generations=-1 \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    ray_init.num_cpus=16 2>&1 | tee $EXPERIMENT_NAME.log
