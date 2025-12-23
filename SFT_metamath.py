import os
import sys
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set this to your desired physical GPU ID

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ==========================================
# 0. CONFIGURATION
# ==========================================
# GPU Setup

# Model & Data Config
# NOTE: Qwen3 is not officially out yet. Ensure this string is valid or use "unsloth/Qwen2.5-7B-Instruct"
MODEL_NAME = "unsloth/Qwen3-4B"  # Change to your target model
# Example HF Dataset (Alpaca style). Change to your target HF dataset.
HF_DATASET_ID = "meta-math/MetaMathQA"
OUTPUT_DIR = "/mnt/ssd/iclrtemp/adapters/qwen3_metamath"
MAX_SEQ_LENGTH = 32768 # Reduced from 32k for stability, increase if VRAM allows
DTYPE = None # None = auto detection (BF16 for Ampere+)
LOAD_IN_4BIT = False # Your script requested False (Max precision), set True for memory savings
MAX_TRAIN_SAMPLES = 50 

# ==========================================
# 1. LOAD MODEL
# ==========================================
print(f"Loading {MODEL_NAME} with Unsloth...")

# Diagnostics
print(f"Visible Devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Torch sees {torch.cuda.device_count()} GPU(s).")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT, 
    device_map = {"": 0} 
)

print("Model on device:", next(model.parameters()).device, "with dtype:", next(model.parameters()).dtype)

# ==========================================
# 2. LORA CONFIGURATION
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================
# 3. PREPARE DATA (HUGGING FACE SOURCE)
# ==========================================
print(f"Loading dataset from Hugging Face: {HF_DATASET_ID}")

# Load dataset from HF Hub
dataset = load_dataset(HF_DATASET_ID, split="train")
original_len = len(dataset)
print(f"Original dataset size: {original_len}")

if MAX_TRAIN_SAMPLES is not None:
    
    # Ensure we don't try to select more than exists
    limit = min(MAX_TRAIN_SAMPLES, original_len)
    print(f"limiting dataset to {limit} examples...")
    dataset = dataset.select(range(limit))

# Adapter function to map HF dataset columns to your script's logic
# Most HF datasets use 'instruction'/'input'/'output' or 'messages'.
# We map them to 'prompt' and 'chosen' to match your formatting function.
def adapt_columns(example):
    # Edit this logic based on the specific HF dataset structure
    prompt = example.get("query", "")
    if example.get("input"):
        prompt += "\n" + example["input"]
    
    output = example.get("response", "")
    
    return {"prompt": prompt, "chosen": output}

# Apply column mapping
dataset = dataset.map(adapt_columns)

# Standard formatting function from your script
def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    responses = examples["chosen"]
    texts = []
    for p, r in zip(prompts, responses):
        # Standard chat structure
        messages = [
            {"role": "user", "content": str(p)},
            {"role": "assistant", "content": str(r)}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts }

print("Formatting dataset...")
dataset = dataset.map(formatting_prompts_func, batched=True)

# test print
print("Sample formatted data:")
print(dataset[0]["text"][:500], "...")

# ==========================================
# 4. TRAINER CONFIGURATION
# ==========================================
print("Configuring Trainer...")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 8,
        
        # Hyperparameters
        warmup_steps = 5,
        max_steps = 500, # Increase this for real training (e.g., 500+)
        learning_rate = 2e-5,
        num_train_epochs = 3, # Set to 3 for full run
        
        # Precision
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        
        # Optimizer
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        
        # Logging
        logging_steps = 1,
        seed = 3407,
        
        # Stability
        max_grad_norm = 1.0, 
    ),
)

# ==========================================
# 5. RUN TRAINING
# ==========================================
print("Starting Training...")
trainer_stats = trainer.train()

print(f"Saving to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ==========================================
# 6. INFERENCE TEST
# ==========================================
print("Running Inference Test...")
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Pick a prompt from the dataset
test_prompt = dataset[0]['prompt']
print(f"Test Prompt: {test_prompt[:100]}...")

messages = [
    {"role": "user", "content": test_prompt}
]

# Note: enable_thinking=True is specific to certain tokenizers/models.
# If using standard Qwen/Llama, this might simply be ignored or need removal.
try:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        enable_thinking=True, 
    ).to("cuda")
except TypeError:
    # Fallback if enable_thinking is not supported by current tokenizer version
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to("cuda")

outputs = model.generate(
    input_ids=inputs, 
    max_new_tokens=32768, 
    use_cache=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

print("="*30)
print("GENERATION OUTPUT:")
print("="*30)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
print("="*30)