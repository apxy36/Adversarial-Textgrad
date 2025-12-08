import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import re
import json
from tqdm import tqdm
# --- Configuration ---
# Path to your fine-tuned adapters (from SFT step) or the merged model
MODEL_PATH = "/mnt/ssd/iclrtemp/adapters/qwen3_gsm8k_manual_loss"
# MODEL_PATH = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit" # Use this to test baseline

# Generation Config
MAX_SEQ_LENGTH = 32768 # Max length for deep reasoning
dtype = None # Auto detection
load_in_4bit = False

# AIME 2025 Data Path (If you have a specific file, set it here)
# If None, it attempts to load standard AIME from HuggingFace
AIME_25_PATH = None 

# --- 1. Robust Answer Extraction Logic ---
def extract_answer_value(text: str) -> str | None:
    """
    Extracts the numerical answer from a solution trace.
    Prioritizes \boxed{}, then "The answer is", then the last number.
    """
    # 1. Check for \boxed{123} (LaTeX style - common in AIME/Math)
    boxed_match = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match[-1].strip() # Return the last boxed value
    
     # Priority 1: Search for all \boxed{...} LaTeX commands.
    # The final answer is the last one.
    try:
        boxed_matches = re.findall(r'\\boxed\{(\d{1,3})\}', text)
        if boxed_matches:
            # Return the last number found inside a \boxed{}
            return boxed_matches[-1]
    except:
        pass
    
    try:
        # Priority 1.5: Search for all [[...]] final answer formats.
        # The final answer is the last one.
        # search for numbers inside [[...]]
        matches = re.findall(r'\[\[(\d{1,4})\]\]', text)
        # find decimals
        if not matches:
            matches = re.findall(r'\[\[(\d{1,4}\.\d+)\]\]', text) # added to catch decimal answers in [[...]]
            # remove decimals from matches via rounding e.g. 123.45 -> 123, 45.00 -> 45, 5.00 -> 5
            if matches:
                rounded_matches = []
                for m in matches:
                    try:
                        rounded_num = str(int(round(float(m))))
                        rounded_matches.append(rounded_num)
                    except:
                        continue
                if rounded_matches:
                    return rounded_matches[-1]
            
        if matches:
            return matches[-1]
    except:
        pass
    # matches = re.findall(r'\[\[(.*?)\]\]', text)
    # if matches:
    #     return matches[-1]

    # 2. Check for explicit text cues
    text_lower = text.lower()
    patterns = [
        r"answer is\s*([0-9\.\-]+)",
        r"answer:\s*([0-9\.\-]+)",
        r"####\s*([0-9\.\-]+)" # GSM8K specific delimiter
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return matches[-1].strip()

    # 3. Fallback: Last number in the text (Risky, but often necessary)
    # Filter out potential year numbers or problem indices if possible, but keep simple here
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    
    return None

def normalize_answer(answer: str) -> float | None:
    """Normalizes string answers to floats for comparison."""
    try:
        # Remove commas, dollar signs, etc.
        clean_ans = re.sub(r"[^\d\.\-]", "", answer)
        return float(clean_ans)
    except (ValueError, TypeError):
        return None

def check_correctness(generated_output, ground_truth):
    extracted_gen = extract_answer_value(generated_output)
    
    # GSM8K often formats ground truth as "reasoning #### 1234"
    if "####" in str(ground_truth):
        ground_truth = str(ground_truth).split("####")[-1].strip()
        
    norm_gen = normalize_answer(extracted_gen)
    norm_gt = normalize_answer(str(ground_truth))
    
    if norm_gen is not None and norm_gt is not None:
        # Floating point comparison with tolerance
        return abs(norm_gen - norm_gt) < 1e-4, extracted_gen
    
    # Fallback to exact string match (for non-numeric answers)
    return str(extracted_gen).strip() == str(ground_truth).strip(), extracted_gen

# --- 2. Evaluation Runner ---
def evaluate_dataset(model, tokenizer, dataset, dataset_name, num_samples=None):
    print(f"\n--- Starting Evaluation: {dataset_name} ---")
    
    correct_count = 0
    total_count = 0
    results_log = []
    
    # Limit samples for quick testing if needed
    eval_data = dataset if num_samples is None else dataset.select(range(min(len(dataset), num_samples)))

    # Enable native inference optimizations
    FastLanguageModel.for_inference(model)

    for i, example in tqdm(enumerate(eval_data), total=len(eval_data)):
        # Handle dataset-specific column names
        if dataset_name == "GSM8K":
            question = example['question']
            ground_truth = example['answer']
        elif "AIME" in dataset_name:
            question = example.get('problem', example.get('question'))
            ground_truth = example.get('solution', example.get('answer'))
            # AIME GT often includes the full proof. We need just the boxed answer or last num.
            ground_truth = extract_answer_value(ground_truth)

        question += "Please put your final answer within \\boxed{}.\n"
        
        # Prepare Prompt (Using Qwen/ChatML format)
        messages = [
            {"role": "user", "content": question}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to("cuda")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_SEQ_LENGTH, # Allow deep thinking
                use_cache=True,
                temperature=0.7, # Greedy decoding for reproducible eval
            )
        
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Isolate the assistant's response (remove the input prompt)
        # Note: formatting depends on tokenizer, usually splitting by "assistant" works
        try:
            response_text = generated_text.split("assistant\n")[-1].strip()
        except:
            response_text = generated_text

        # Verify
        is_correct, extracted_val = check_correctness(response_text, ground_truth)
        
        if is_correct:
            print(f"Sample {i}: Correct")
            correct_count += 1
        else:
            print(f"Sample {i}: Incorrect")
            print(f"  Question: {question}")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Generated Response: {response_text}")
            print(f"  Extracted Answer: {extracted_val}")
        total_count += 1
        
        results_log.append({
            "question": question,
            "ground_truth": str(ground_truth),
            "generated_response": response_text,
            "extracted_answer": str(extracted_val),
            "is_correct": is_correct
        })

    accuracy = correct_count / total_count
    print(f"\nResults for {dataset_name}:")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    return accuracy, results_log

# --- 3. Main Execution ---
def main():
    # Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map = {"": 0}
    )
    
    # 1. Evaluate GSM8K (Standard Test Set)
    # print("Loading GSM8K...")
    # gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    # acc_gsm, log_gsm = evaluate_dataset(model, tokenizer, gsm8k, "GSM8K", num_samples=100) # Set num_samples for debugging
    
    # 2. Evaluate AIME 2025 (Or fallback to standard AIME)
    print("Loading AIME...")
    if AIME_25_PATH and os.path.exists(AIME_25_PATH):
        # Load local JSONL file for specific AIME 2025 data
        aime_dataset = load_dataset("json", data_files=AIME_25_PATH, split="train")
        dataset_label = "AIME 2025 (Local)"
    else:
        # Fallback to HuggingFace AIME dataset (usually up to 2023/24)
        # We take a slice of the end to approximate "hard/recent" problems
        print("AIME 25 file not found/provided. Loading standard AIME from Hub.")
        full_aime = load_dataset("math-ai/aime25", "default", split="test")
        # Take the last 50 problems as a proxy for 'recent/hard'
        aime_dataset = full_aime.select(range(0, len(full_aime))) 
        dataset_label = "AIME (Last 50 Samples)"

    acc_aime, log_aime = evaluate_dataset(model, tokenizer, aime_dataset, dataset_label)

    # Save Results
    output_file = "evaluation_results_2.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {"model": MODEL_PATH},
            # "gsm8k": {"accuracy": acc_gsm, "details": log_gsm},
            "aime": {"accuracy": acc_aime, "details": log_aime}
        }, f, indent=2)
    print(f"Full logs saved to {output_file}")

if __name__ == "__main__":
    main()