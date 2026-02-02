ACTOR_PATH=""
TARGET_PATH=""

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $ACTOR_PATH \
    --target_dir $TARGET_PATH
