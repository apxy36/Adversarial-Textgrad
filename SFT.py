import os, sys, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
# import os

# Config
MODEL_NAME = "unsloth/Qwen3-4B"
INPUT_DATASET = "SFTpairs6.jsonl"
OUTPUT_DIR = "/mnt/ssd/iclrtemp/adapters/qwen3_gsm8k_v2.6"
MAX_SEQ_LENGTH = 32768




# ==========================================
# 0. CONFIGURATION
# ==========================================
# If "unsloth/Qwen3-4B" continues to fail, change this to "unsloth/Qwen2.5-3B-Instruct"
# to rule out the specific model file being broken.

# ==========================================
# 1. LOAD MODEL (The Unsloth Way)
# ==========================================
print(f"Loading {MODEL_NAME} with Unsloth...")

# Verify we are on the right GPU
print(f"Visible Devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Torch sees {torch.cuda.device_count()} GPU(s).")
TARGET_GPU = 0  # Change this to your target GPU ID
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,           # Auto-detects BF16 for RTX 6000
    load_in_4bit = False,   # Set True if you want to save VRAM, False for max precision
    device_map = {"": 0} 
)

print("Model on device:", next(model.parameters()).device, "with dtype:", next(model.parameters()).dtype)

# 2. LoRA CONFIGURATION
# We use standard settings. No manual alpha tweaking or resizing.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Use Unsloth's optimized checkpointing
    random_state = 3407,

)

# ==========================================
# 3. PREPARE DATA
# ==========================================
# Simple formatting function matching the chat template
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

dataset = load_dataset("json", data_files=INPUT_DATASET, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# ==========================================
# 4. TRAINER CONFIGURATION
# ==========================================
# We use standard TrainingArguments. 
# We explicitly set gradient_checkpointing to False if you are seeing NaNs,
# but "unsloth" usually handles it well. 
# IF THIS FAILS: Change `per_device_train_batch_size` to 1.

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # Packing can sometimes cause stability issues on new models
    args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 8,
        
        # Hyperparameters
        warmup_steps = 5,
        max_steps = 500,
        learning_rate = 2e-5, # Standard safe LR
        num_train_epochs = 3,
        
        # Precision (Native BF16 for RTX 6000)
        fp16 = False,
        bf16 = True,
        
        # Optimizer
        optim = "adamw_torch", # Unsloth standard
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        
        # Logging
        logging_steps = 1,
        seed = 3407,
        
        # Stability settings
        max_grad_norm = 1.0, 
    ),
)

# ==========================================
# 5. RUN
# ==========================================
print("Starting Training...")
trainer_stats = trainer.train()

print(f"Saving to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# Pick a prompt from the dataset
test_prompt = dataset[0]['prompt'] # Original prompt
messages = [
    {"role": "user", "content": test_prompt}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    truncation=True,            # <--- ADD THIS
    max_length=MAX_SEQ_LENGTH,   # <--- ADD THIS (ensure it matches training)
    enable_thinking=True,
).to(f"cuda:0" )

outputs = model.generate(
    input_ids=inputs, 
    max_new_tokens=1024, 
    use_cache=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id # <--- Explicitly set pad token to suppress warning
)
print(tokenizer.batch_decode(outputs)[0])