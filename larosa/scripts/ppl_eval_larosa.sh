export HF_ENDPOINT=https://hf-mirror.com

#qwen
CUDA_VISIBLE_DEVICES=1 python3 test/ppl_test_larosa_qwen.py --model_name Qwen2.5-7B-larosa --larosa_path qwen2_7b_act_rotate_mlp_and_attn_seperate_Q_wikitext_opensource --sparsity 0.5

#llama
CUDA_VISIBLE_DEVICES=1 python3 test/ppl_test_larosa_llama.py --model_name llama-3-8b-larosa --larosa_path llama2_act_rotate_mlp_and_attn_seperate_Q --sparsity 0.5