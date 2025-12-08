import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Config
MODEL_NAME = "Qwen/Qwen3-4B"
INPUT_DATASET = "SFTpairs3.jsonl"

print(f"--- Loading {MODEL_NAME} for Deep Diagnostics ---")

# 1. Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto", # Match your training setup
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# 2. WEIGHT CHECK: Are the weights already broken?
print("\n--- Step 1: Checking Model Weights for NaNs ---")
has_nan_weights = False
for name, param in model.named_parameters():
    if torch.isnan(param).any() or torch.isinf(param).any():
        print(f"CRITICAL: Found NaN/Inf in parameter: {name}")
        has_nan_weights = True

if has_nan_weights:
    print("CONCLUSION: The model file itself is corrupted. You must download a different checkpoint.")
    exit()
else:
    print("PASSED: Initial weights are healthy.")

# 3. HOOKS: Register probes to catch NaNs during calculation
print("\n--- Step 2: Registering Layer Probes ---")

def check_nan_hook(module, args, output):
    # Output can be a tuple or a tensor
    if isinstance(output, tuple):
        tensor = output[0]
    else:
        tensor = output
    
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any():
            print(f"!!! NAN DETECTED in output of layer: {module.__class__.__name__} !!!")
            print(f"    Layer info: {module}")
            raise RuntimeError("NaN Found")
        elif torch.isinf(tensor).any():
            print(f"!!! INF DETECTED in output of layer: {module.__class__.__name__} !!!")
            raise RuntimeError("Inf Found")

# Attach hooks to leaf modules (Linear, Norm, Attention)
for name, module in model.named_modules():
    # We probe Linear and Norm layers as they are the most likely to explode
    if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) or "Norm" in name or "Attention" in name:
        module.register_forward_hook(check_nan_hook)

# 4. RUN FORWARD PASS
print("\n--- Step 3: Running Forward Pass with Data ---")
dataset = load_dataset("json", data_files=INPUT_DATASET, split="train")
sample_text = dataset[0]["prompt"] + dataset[0]["chosen"]

inputs = tokenizer(
    sample_text, 
    return_tensors="pt", 
    max_length=2048, 
    truncation=True
).to("cuda")

try:
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
    print("\nPASSED: Forward pass completed without NaNs.")
    print(f"Final Loss Calculation: {outputs.loss if hasattr(outputs, 'loss') else 'N/A'}")
except RuntimeError as e:
    if "NaN Found" in str(e) or "Inf Found" in str(e):
        print("\nDIAGNOSIS COMPLETE.")
    else:
        print(f"\nCRASHED with unrelated error: {e}")