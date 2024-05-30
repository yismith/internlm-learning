from huggingface_hub import hf_hub_download  # Load model directly

if __name__ == '__main__':
    hf_hub_download(repo_id="internlm/internlm2-7b", filename="config.json", local_dir="./")
