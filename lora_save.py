import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Adjust as needed
# 1. Define paths
adapter_path = "/mnt/ssd/iclrtemp/adapters/qwen3_14b_MATH600_SFT_3e-7/checkpoint-225"
output_path = "/mnt/ssd/iclrtemp/adapters/qwen3_14b_MATH600_SFT_3e-7/merged_model"

# 2. Load Base Model (Use the one found in your adapter_config.json)
# Assuming it's Qwen based on your folder name, but check adapter_config.json to be sure.
base_model_id = "Qwen/Qwen3-14B"  # <--- REPLACE THIS WITH YOUR ACTUAL BASE MODEL

print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 3. Load Adapter
print(f"Loading LoRA adapter: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path, dtype=torch.float16)

# 4. Merge
print("Merging weights...")
model = model.merge_and_unload()

# 5. Save Full Model
print(f"Saving merged model to: {output_path}")
model.save_pretrained(output_path)

# 6. Save Tokenizer (Crucial for vLLM)
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

print("Done! Point your vLLM script to the output_path.")