export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_PATH="Qwen2.5-7B-larosa"
OUTPUT_PATH="lm_eval_output/"

# --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
accelerate launch -m --main_process_port 12329 lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,boolq \
    --batch_size 12 \
    --log_samples \
    --write_out \
    --output_path $OUTPUT_PATH