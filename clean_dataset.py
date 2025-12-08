import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import json
import os 
import numpy as np

# Configuration
MODEL_NAME = "unsloth/Qwen3-4B"
INPUT_FILE = "SFTpairs3.jsonl"
FILTERED_FILE = "SFTpairs3_filtered.jsonl"
MAX_SEQ_LENGTH = 2048

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
vocab_limit = config.vocab_size
print(f"Original Model Vocab Size: {vocab_limit}")

# 1. Define Formatting
def format_text(prompt, chosen):
    # Use standard text format without special control tokens first
    return f"User: {prompt}\nAssistant: {chosen}"

# 2. Filter Loop
valid_count = 0
dropped_count = 0

with open(INPUT_FILE, 'r') as fin, open(FILTERED_FILE, 'w') as fout:
    for line in fin:
        try:
            data = json.loads(line)
            text = format_text(data['prompt'], data['chosen'])

            # search for boxed{ in data['chosen'] and replace with \boxed{
            text = text.replace("boxed{", "\\boxed{")
            
            # Tokenize
            ids = tokenizer.encode(text, add_special_tokens=False)
            
            # CHECK: Are all IDs within the original vocab?
            if max(ids) >= vocab_limit:
                dropped_count += 1
                continue # Skip this example
                
            # If safe, write it
            fout.write(line)
            valid_count += 1
        except Exception:
            dropped_count += 1

print(f"Filtering Complete.")
print(f"Retained: {valid_count}")
print(f"Dropped:  {dropped_count} (Contained OOB tokens)")

# import os
# from datasets import load_dataset
# from transformers import AutoTokenizer
# import numpy as np

# --- Configuration ---
# Path to your existing dataset (from Stage 2)
INPUT_DATASET = "SFTpairs4.jsonl"
# Where to save the cleaned version
OUTPUT_DATASET = "SFTpairs4_cleaned.jsonl"
# The model you intend to train (needed to load the correct tokenizer/chat template)
MODEL_NAME = "unsloth/Qwen3-4B"

# The maximum sequence length you plan to use in SFT (e.g., 2048, 4096, 8192)
MAX_SEQ_LENGTH = 2048

def main():
    print(f"Loading tokenizer for {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading dataset from {INPUT_DATASET}...")
    dataset = load_dataset("json", data_files=INPUT_DATASET, split="train")
    original_size = len(dataset)
    print(f"Original dataset size: {original_size}")

    # --- Length Calculation Function ---
    def calculate_length(example):
        """
        Formats the input exactly how the SFTTrainer will see it
        (User Prompt + Assistant Response + Special Tokens) and counts tokens.
        """
        # Construct the conversation structure
        # Note: If your dataset keys are different (e.g., 'instruction'/'output'), adjust here.
        messages = [
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": example['chosen']}
        ]
        
        # apply_chat_template adds <|im_start|>, <|im_end|>, newlines, etc.
        # This is crucial for an accurate count.
        tokenized_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False
        )
        
        return {"total_tokens": len(tokenized_ids)}

    # Map the length calculation across the dataset (multiprocessed for speed)
    print("Calculating token lengths...")
    dataset_with_lengths = dataset.map(calculate_length, num_proc=os.cpu_count())

    # --- Statistics ---
    lengths = dataset_with_lengths["total_tokens"]
    max_len_found = np.max(lengths)
    avg_len_found = np.mean(lengths)
    
    print(f"\n--- Statistics ---")
    print(f"Target Max Length: {MAX_SEQ_LENGTH}")
    print(f"Longest Example Found: {max_len_found}")
    print(f"Average Length: {avg_len_found:.2f}")

    # --- Filtering ---
    if max_len_found > MAX_SEQ_LENGTH:
        print(f"\nExamples exceeding {MAX_SEQ_LENGTH} tokens detected. Filtering...")
        
        # Filter the dataset
        cleaned_dataset = dataset.filter(lambda x: calculate_length(x)["total_tokens"] <= MAX_SEQ_LENGTH)
        
        removed_count = original_size - len(cleaned_dataset)
        print(f"Removed {removed_count} examples.")
        print(f"New dataset size: {len(cleaned_dataset)}")
        
        # Save the clean version
        print(f"Saving cleaned dataset to {OUTPUT_DATASET}...")
        cleaned_dataset.to_json(OUTPUT_DATASET)
        print("Done.")
    else:
        print("\nNo examples exceeded the max sequence length. No changes needed.")
        # Optionally save it anyway so the filename is consistent for the next step
        dataset.to_json(OUTPUT_DATASET)

if __name__ == "__main__":
    main()