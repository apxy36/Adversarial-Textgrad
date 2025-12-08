import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import numpy as np

# --- Configuration ---
MODEL_NAME = "unsloth/Qwen3-4B"
INPUT_DATASET = "SFTpairs4.jsonl"
MAX_SEQ_LENGTH = 2048

print(f"--- Diagnosing Tokenizer & Data for {MODEL_NAME} ---")

# 1. Load Config & Tokenizer
config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Get the absolute hardware limit for token IDs
vocab_limit = config.vocab_size
print(f"Model Vocab Size: {vocab_limit}")
print(f"Tokenizer Vocab Size: {tokenizer.vocab_size}")

# 2. Define Formatting (Same as your training script)
def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    responses = examples["chosen"]
    texts = []
    for prompt, response in zip(prompts, responses):
        # Manual Qwen ChatML format to ensure control
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        texts.append(text)
    return { "text" : texts }

# 3. Load Data
dataset = load_dataset("json", data_files=INPUT_DATASET, split="train")
print(f"Loaded {len(dataset)} examples from {INPUT_DATASET}")
print("Example entry:", dataset[0]['prompt'])
print("Example chosen response:", dataset[0]['chosen'])
print("Example rejected response:", dataset[0]['rejected'])
dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"Scanning {len(dataset)} examples...")

# 4. Scan for Errors
max_found_id = 0
found_empty_labels = 0
found_oob_tokens = 0

for i, example in enumerate(dataset):
    # Tokenize without padding first to check raw IDs
    tokens = tokenizer(example["text"], truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids = tokens["input_ids"]
    
    # Check 1: Empty input
    if len(input_ids) == 0:
        print(f"⚠️ Error: Example {i} resulted in 0 tokens!")
        continue

    # Check 2: Out of Bounds Tokens
    current_max = max(input_ids)
    if current_max > max_found_id:
        max_found_id = current_max
    
    if current_max >= vocab_limit:
        print(f"🚨 CRITICAL ERROR at Index {i}:")
        print(f"   Found Token ID {current_max} which is >= Model Vocab Limit {vocab_limit}")
        print(f"   Text snippet: {example['text'][:100]}...")
        found_oob_tokens += 1
        break # Stop immediately if we find the smoking gun

    # Check 3: Simulating Label Creation (Data Collator Logic)
    # This logic mimics how SFTTrainer masks user prompts.
    # If the logic fails and everything becomes -100, loss is NaN.
    
    # Find the start of the assistant response
    # Qwen assistant header token IDs usually end with <|im_start|>assistant...
    # For Qwen 2.5: <|im_start|> (151644) assistant (77091) \n (198)
    
    # A rough check: does the tokenized sequence contain the assistant header components?
    # If using DataCollatorForCompletionOnlyLM, this usually breaks.
    # Since we are doing standard SFT on the whole sequence in the previous script,
    # we just need to ensure the sequence isn't empty.
    pass

print("\n--- Diagnostic Results ---")
print(f"Max Token ID found in dataset: {max_found_id}")
print(f"Model Vocab Limit:             {vocab_limit}")

if max_found_id >= vocab_limit:
    print("\n❌ DIAGNOSIS: TOKEN ID OVERFLOW")
    print("The tokenizer is producing IDs that the 4-bit model weights do not have entries for.")
    print("This causes immediate NaNs because the model looks up index X in a matrix of size Y (where X > Y).")
elif found_oob_tokens == 0:
    print("\n✅ DIAGNOSIS: Tokens look valid.")
    print("The NaN is likely coming from the Embedding Layer dtype or Gradient Scaling.")