from init import *
import os, sys, argparse

from huggingface_hub import HfApi

api = HfApi()

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the adapter model is saved.", default="/mnt/ssd/iclrtemp/adapters/qwen3_14b_MMIQC800_SFT_3e-7")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID to upload the model to.")
    args = parser.parse_args()
    # Upload the folder containing adapter_model.safetensors and adapter_config.json
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo_id,
        repo_type="model",
    )

if __name__ == "__main__":
    main()