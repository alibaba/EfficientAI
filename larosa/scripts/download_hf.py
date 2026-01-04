import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
from datasets import load_dataset

# download Qwen2.5
snapshot_download(repo_id="Qwen/Qwen2.5-7B", local_dir="Qwen2.5-7B")