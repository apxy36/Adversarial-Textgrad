import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_NAME = "unsloth/Qwen3-4B" 
INPUT_DATASET = "SFTpairs4.jsonl"

print(f"--- ISOLATION TEST: Loading {MODEL_NAME} with Pure Transformers ---")

# 1. Load Model with Standard Transformers (No Unsloth)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16, # Use BF16 natively
        device_map="cuda",
        trust_remote_code=True 
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
except Exception as e:
    print(f"FAILED to load model: {e}")
    exit()

# 2. Fix Padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Create a Dummy Input
test_text = "Hello, how are you?"
inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

# 4. Run Forward Pass
print("Running forward pass...")
with torch.no_grad():
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        labels=inputs.input_ids # Calculate loss
    )

print(f"Loss: {outputs.loss}")

if torch.isnan(outputs.loss):
    print("RESULT: FAILURE. The model itself produces NaNs. The weights are likely corrupted or RoPE scaling is wrong in config.json.")
else:
    print("RESULT: SUCCESS. Standard Transformers works. The issue is inside Unsloth's kernels.")