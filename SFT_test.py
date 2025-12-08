import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Using a known stable model to rule out architecture bugs.
# If you must use "unsloth/Qwen3-4B", change it here, but Qwen2.5 is the stable standard.
MODEL_ID = "Qwen/Qwen3-4B" 
DATA_FILE = "SFTpairs3.jsonl"
OUTPUT_DIR = "./output_clean_run"
MAX_SEQ_LENGTH = 2048

# ==========================================
# 1. LOAD MODEL & TOKENIZER
# ==========================================
print(f"Loading model: {MODEL_ID}")
# Load in bfloat16 (Standard for RTX 6000 Ada)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    device_map="cuda:1",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Required: Qwen models usually don't have a default pad token for training
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Standard for training

# ==========================================
# 2. SETUP LORA
# ==========================================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==========================================
# 3. PREPARE DATASET (Standard Map)
# ==========================================
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def preprocess_function(example):
    # 1. Create text
    # Adjust this line if your JSONL keys are different
    prompt = example.get("prompt", "")
    response = example.get("chosen", "") 
    full_text = f"User: {prompt}\nAssistant: {response}<|endoftext|>"
    
    # 2. Tokenize
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False # Padding handled by DataCollator
    )
    
    # 3. Add Labels (Standard Causal LM training: labels == input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

# ==========================================
# 4. TRAINING ARGUMENTS
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # Safe batch size for 4B model
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=1,
    bf16=True,  # Enable BF16
    fp16=False,
    optim="adamw_torch", # Standard PyTorch optimizer
    save_strategy="no",
    report_to="none",
    # Defaults
    weight_decay=0.01,
    max_grad_norm=1.0,
    gradient_checkpointing=False # Disabled for stability
)

# ==========================================
# 5. INITIALIZE TRAINER
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

# ==========================================
# 6. RUN
# ==========================================
print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)