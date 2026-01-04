export HF_DATASETS_OFFLINE=1
export HF_ENDPOINT="https://hf-mirror.com"
CUDA_VISIBLE_DEVICES=1 python3 gen_act/grab_acts_rotate_diff_Q_qwen.py --model_name Qwen2.5-7B-larosa --output_path qwen2_7b_act_rotate_mlp_and_attn_seperate_Q_wikitext_opensource