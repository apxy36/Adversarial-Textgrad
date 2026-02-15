import argparse
import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from unsloth import FastLanguageModel
from math_verify import parse
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
math_perturb_dir = os.path.join(current_dir, 'eval', "MATH_Perturb")
print("MATH perturb dir: " + math_perturb_dir)

# 2. Add it to Python's search path
if math_perturb_dir not in sys.path:
    sys.path.append(math_perturb_dir)
from eval.MATH_Perturb.evaluate_perturb import extract_ground_truth_answer
BATCH_SIZE = 64
# --- 1. Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Validation Pipeline with Optional Plot Saving")
    parser.add_argument("--seed_file", type=str, required=True, help="Path to original/seed dataset JSON/JSONL")
    parser.add_argument("--hardened_file", type=str, required=True, help="Path to hardened adversarial dataset JSON/JSONL")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-14B-Instruct-bnb-4bit")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of samples to test")
    
    # --- NEW: Optional Plot Saving ---
    parser.add_argument("--save_plot_path", type=str, default=None, 
                        help="Filename to save the graph (e.g. 'confidence_shift.png'). If not provided, graph is not saved.")
    parser.add_argument("--save_json_path", type=str, default=None, help="Path to save raw results for external graphing")
    return parser.parse_args()

# --- 2. Robust Data Loader (From previous steps) ---
def extract_sample_list(data):
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            keys = data[0].keys()
            if any(k in keys for k in ['question', 'prompt', 'ground_truth', 'generated_response', 'details']):
                return data
        return []
    if isinstance(data, dict):
        priority_keys = ['details', 'samples', 'data', 'generations', 'results']
        for key in priority_keys:
            if key in data:
                found = extract_sample_list(data[key])
                if found: return found
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                found = extract_sample_list(value)
                if found: return found
    return []

def load_data(filepath, max_samples, seed=True):
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char in ['[', '{']:
            try:
                raw = json.load(f)
                data = extract_sample_list(raw)
            except: data = []
        else:
            try:
                data = [json.loads(line) for line in f if line.strip()]
            except: data = []
    
    # Filter for valid entries immediately
    valid_data = []
    for item in data:
        if seed:
            p = item.get('prompt') or item.get('question') or item.get('problem')
            # a = item.get('ground_truth') or item.get('answer') or item.get('solution') or item.get('original_GT')
        else:
            p = item.get('prompt')
        # p = item.get('prompt') or item.get('question') or item.get('problem')
        a = item.get('ground_truth') or item.get('answer') or item.get('solution') or item.get('original_GT')
        if p and a:
            valid_data.append({'p': p, 'a': a})
            
    if len(valid_data) > max_samples:
        return valid_data[:max_samples]
    return valid_data

def extract_answer(response_str):
    try:
        parsed = parse(response_str)
        if len(parsed['answers']) > 0:
            return parsed['answers'][0]
    except:
        pass
    try:
        parsed = extract_ground_truth_answer("", response_str, dataset_type='original')
        if parsed:
            return parsed
        
    except:
        pass
    return None


training_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

                Instruction:
                Solve the following problem.
                - Provide a direct final step-by-step solution in a single pass.
                - Put your final answer on its own line as \\boxed{{...}}.
                - Use exactly one \\boxed{{}} and do not box intermediate results.
                
                Output format:
                -Steps: (multiple lines)
                -Final answer: \\boxed{{...}}  (last line only)"""
# --- FAST BATCHED INFERENCE ---



def get_answer_probe_probs(model, tokenizer, dataset):
    probs = []
    
    # Batching logic
    batches = [dataset[i:i + BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]
    
    print(f"Probing {len(dataset)} samples...")
    
    # Pre-calculate the trigger phrase tokens
    # e.g., "\n\nTherefore, the final answer is"
    trigger_text = "\n\nTherefore, final answer:"

    for item in tqdm(dataset):
    
        prompt = item['p']
        answer = str(item['a']).strip()
        extracted_answer = extract_answer(answer)

        delimiter = "Output format:\n-Steps: (multiple lines)\n-Final answer: \\boxed{...}  (last line only)\n\n"

        if delimiter in prompt:
            # split()[1] takes everything AFTER the delimiter
            output_str = prompt.split(delimiter)[1]
            print("Delimiter found. Extracted relevant part of the string.")
        else:
            output_str = prompt
            print("Delimiter not found. Returning original string.")
        extracted_problem = output_str.strip()
        
        # 1. Construct a "Forced" context
        # We pretend the model has already reasoned and is about to say the number.
        forced_input_text = f"{extracted_problem} {trigger_text} {extracted_answer}"

        context_text = f"{prompt}{trigger_text}"
        
        # 2. Tokenize Context (without special tokens to avoid EOS/BOS confusion)
        context_enc = tokenizer(context_text, return_tensors="pt", add_special_tokens=False)
        context_ids = context_enc.input_ids.to("cuda")
        context_len = context_ids.shape[1]
        
        # 3. Define Full Text (Context + Answer)
        # We add a space before the answer to handle tokenization correctly
        clean_answer = str(answer).strip()
        full_text = f"{context_text} {clean_answer}"

        val_token = str(extracted_answer).split()[0]
        # ids = tokenizer.encode(val_token, add_special_tokens=False)

        # 2. Tokenize
        # We assume single batch size for this specific function logic

        full_ids = tokenizer.apply_chat_template(   
            [
                {"role": "system", "content": training_prompt},
                {"role": "user", "content": full_text}
            ],
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
            truncation=True,
        ).to("cuda")

        if full_ids.shape[1] <= context_len:
            continue

        target_token_id = full_ids[0, context_len].item()
        
        # 5. Forward Pass
        # We only need to run the model up to the point where it predicts the answer
        # We pass the context + the target token (so we can get the logit for the target)
        # We use input_ids up to `context_len + 1`
        pass_ids = full_ids[:, :context_len + 1]
        
        with torch.no_grad():
            outputs = model(pass_ids)
            
            # 6. Extract Logits
            # We want the prediction made *at the end of the context*
            # This corresponds to the logit at index `context_len - 1`
            # (because logit[i] is the prediction for token[i+1])
            logits = outputs.logits[0, context_len - 1, :]
            
            # 7. Calculate Probability
            prob = torch.softmax(logits, dim=-1)[target_token_id].item()
            probs.append(prob)

        # encoded = tokenizer.apply_chat_template(
        #     [{"role": "system", "content": training_prompt},
        #         {"role": "user", "content": forced_input_text}],
        #     return_tensors="pt",
        #     add_generation_prompt=True,
        #     enable_thinking=False,
        #     truncation=True,
        # ).to("cuda")

        # if hasattr(encoded, "input_ids"):
        #     input_ids = encoded.input_ids
        # else:
        #     input_ids = encoded
        # # input_ids = encoded.input_ids
        
        # # 3. Find the index of the Answer Token
        # # It is the last token in the sequence
        # target_token_id = input_ids[0, -1]
        
        # # 4. Forward Pass
        # with torch.no_grad():
        #     # FIX IS HERE: Pass input_ids directly, OR pass (**encoded)
        #     # Since we extracted input_ids above, we pass it directly.
        #     outputs = model(input_ids)
            
        #     # Look at the logits of the token *before* the answer
        #     logits = outputs.logits[0, -2, :] 
            
        #     # 5. Calculate Probability
        #     prob = torch.softmax(logits, dim=-1)[target_token_id].item()
        #     probs.append(prob)
            

            
    return probs

def get_batch_probs(model, tokenizer, dataset):
    probs = []
    
    # Create batches
    batches = [dataset[i:i + BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]
    
    print(f"Processing {len(dataset)} samples in {len(batches)} batches...")
    
    for batch in tqdm(batches):
        prompts = [item['p'] for item in batch]
        answers = [item['a'] for item in batch]
        
        # 1. Prepare Target Variations (CPU side)
        # We need to find the token ID for the answer.
        # Problem: "5" and " 5" are different tokens.
        target_candidates = [] # List of (index_in_batch, [candidate_id_1, candidate_id_2])
        valid_indices = []
        
        for i, ans in enumerate(answers):
            # Extract first word
            first_word = str(ans).strip().split()[0]
            
            # Candidate 1: Raw word (e.g. "5")
            ids_raw = tokenizer.encode(first_word, add_special_tokens=False)
            
            # Candidate 2: Word with leading space (e.g. " 5")
            # This is critical for Llama/Qwen tokenizers
            ids_space = tokenizer.encode(" " + first_word, add_special_tokens=False)
            
            candidates = []
            if ids_raw: candidates.append(ids_raw[0])
            if ids_space: candidates.append(ids_space[0])
            
            if candidates:
                target_candidates.append(list(set(candidates))) # Unique candidates
                valid_indices.append(i)
        
        if not valid_indices: continue

        # Filter prompts
        active_prompts = [prompts[i] for i in valid_indices]
        
        # 2. Tokenize with Chat Template
        # Ensure add_generation_prompt=True is handled correctly by the tokenizer
        # We manually apply template to control the end string if needed
        formatted_prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False, truncation=True)
            for p in active_prompts
        ]
        
        inputs = tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to("cuda")
        
        # 3. Fast Forward Pass
        with torch.no_grad():
            outputs = model(**inputs)
            # Logits of the NEXT token (last position)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # 4. Extract Max Probability across Candidates
        # If "5" is 0.01 and " 5" is 0.99, we want 0.99.
        for i, candidates in enumerate(target_candidates):
            # Get the probabilities for all candidate tokens for this sample
            candidate_probs = [next_token_probs[i, tid].item() for tid in candidates]
            
            # Take the maximum probability found among valid variations
            if candidate_probs:
                probs.append(max(candidate_probs))
            else:
                probs.append(0.0)
            
    return probs

def get_batch_probs2(model, tokenizer, dataset):
    probs = []
    
    # Create batches
    batches = [dataset[i:i + BATCH_SIZE] for i in range(0, len(dataset), BATCH_SIZE)]
    
    print(f"Processing {len(dataset)} samples in {len(batches)} batches...")
    
    for batch in tqdm(batches):
        prompts = [item['p'] for item in batch]
        answers = [item['a'] for item in batch]
        
        # 1. Prepare Target Token IDs (CPU side)
        target_ids = []
        valid_indices = [] # Track which items in batch are valid
        
        for i, ans in enumerate(answers):
            # Get the first significant token of the answer
            # We strip whitespace to avoid issues with leading spaces
            first_word = str(ans).strip().split()[0]
            ids = tokenizer.encode(first_word, add_special_tokens=False)
            if ids:
                target_ids.append(ids[0])
                valid_indices.append(i)
        
        if not valid_indices: continue

        # Filter prompts to match valid targets
        active_prompts = [prompts[i] for i in valid_indices]
        
        # 2. Batch Tokenization (Padding on Left is critical for finding last token)
        inputs = tokenizer(
            active_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to("cuda")
        
        # 3. Fast Forward Pass
        with torch.no_grad():
            outputs = model(**inputs)
            # Grab logits of the LAST token in the sequence
            # [Batch, Seq, Vocab] -> [Batch, Vocab]
            next_token_logits = outputs.logits[:, -1, :]
            
            # Softmax to get probabilities
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # 4. Extract specific target probabilities
        # We start moving things to CPU only here to save GPU sync time
        for i, t_id in enumerate(target_ids):
            prob = next_token_probs[i, t_id].item()
            probs.append(prob)
            
    return probs

def get_sequence_confidence(model, tokenizer, prompt, answer):
    """
    Calculates the geometric mean of the probabilities assigned to 
    the tokens in the answer string.
    """
    # 1. Format the Prompt (Context)
    # We use add_generation_prompt=True to get the <|im_start|>assistant header
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # 2. Tokenize Prompt and Full Sequence (Prompt + Answer)
    # We need to know exactly where the answer starts to mask out the prompt
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")
    prompt_len = prompt_ids.shape[1]
    
    # Clean the answer string
    clean_answer = str(answer).strip()
    
    # Full Sequence: Prompt + Answer
    # We append the answer directly to the prompt template
    full_text = prompt_text + clean_answer
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to("cuda")
    
    # Safety Check: If tokenization failed or answer is empty
    if full_ids.shape[1] <= prompt_len:
        return None

    # 3. Forward Pass
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits # Shape: [1, seq_len, vocab_size]

    # 4. Calculate Log-Probabilities
    # We shift logits so that token [i] predicts token [i+1]
    # logits[:, :-1, :] are the predictions
    # full_ids[:, 1:] are the targets
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    
    # Calculate Cross Entropy (Negative Log Likelihood) per token
    # reduction='none' gives us the loss for every specific token
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)), 
        shift_labels.reshape(-1)
    )
    
    # 5. Isolate the Answer Tokens
    # The labels corresponding to the answer start at index (prompt_len - 1)
    # in the shifted vector because label[i] corresponds to logit[i]
    answer_losses = token_losses[prompt_len - 1 :]
    
    if len(answer_losses) == 0: return None

    # 6. Compute Metric: Average Token Probability
    # We convert NLL back to probability: P = exp(-NLL)
    # We take the mean NLL first to normalize for length
    mean_nll = answer_losses.mean()
    confidence = torch.exp(-mean_nll).item()
    
    return confidence

# --- Updated get_probs wrapper ---
def get_probs(model, tokenizer, data, seed=False):
    probs = []
    print(f"Calculating sequence confidence for {len(data)} samples...")
    
    for item in tqdm(data):
        if seed:
            prompt = item.get('prompt') or item.get('question') or item.get('problem')
            # full_answer = item.get('ground_truth') or item.get('answer') or item.get('solution') or item.get('original_GT')
        else:
            prompt = item.get('prompt')
        full_answer = item.get('ground_truth') or item.get('answer') or item.get('solution') or item.get('original_GT')
        
        if not prompt or not full_answer: continue
        
        # We pass the FULL answer now, not just the first token
        p = get_sequence_confidence(model, tokenizer, prompt, full_answer)
        
        if p is not None:
            probs.append(p)
            
    return probs

# --- 3. Probability Logic ---
def get_probs2(model, tokenizer, data):
    probs = []
    print(f"Calculating confidence for {len(data)} samples...")
    for item in tqdm(data):
        prompt = item.get('prompt') or item.get('question') or item.get('problem')
        # Robust answer key search
        full_answer = item.get('ground_truth') or item.get('answer') or item.get('solution') or item.get('original_GT')
        
        if not prompt or not full_answer: continue
        
        # Target the first token of the answer for zero-shot confidence check
        target_token = str(full_answer).strip().split()[0] 
        
        # Tokenize
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        target_ids = tokenizer.encode(target_token, add_special_tokens=False)
        
        if not target_ids: continue
        target_id = target_ids[0]

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs_tensor = torch.softmax(logits, dim=-1)
            probs.append(probs_tensor[target_id].item())
            
    return probs

# --- 4. Main Execution ---
def main():
    args = parse_args()
    
    # Load Model
    print(f"Loading Model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)
    
    # Load Data
    seed_data = load_data(args.seed_file, args.max_samples, seed=True)
    hard_data = load_data(args.hardened_file, args.max_samples, seed=False)
    
    # Get Probabilities
    seed_probs = get_answer_probe_probs(model, tokenizer, seed_data)
    hard_probs = get_answer_probe_probs(model, tokenizer, hard_data)
    
    if not seed_probs or not hard_probs:
        print("Error: Could not extract probabilities. Check dataset format.")
        return

    # Statistics
    u_stat, p_value = stats.mannwhitneyu(seed_probs, hard_probs, alternative='greater')

    mean_seed = np.mean(seed_probs)
    mean_hard = np.mean(hard_probs)
    print(f"\nMean Confidence (Seed): {np.mean(seed_probs):.4f}")
    print(f"Mean Confidence (Hardened): {np.mean(hard_probs):.4f}")
    print(f"P-Value: {p_value:.5e}")

    # --- 5. Visualization & Saving ---
    plt.figure(figsize=(10, 6))
    
    # Plotting
    sns.kdeplot(seed_probs, fill=True, label='Original Dataset', color='blue', alpha=0.3, clip=(0,1))
    sns.kdeplot(hard_probs, fill=True, label='Hardened Dataset', color='red', alpha=0.3, clip=(0,1))
    
    plt.axvline(np.mean(seed_probs), color='blue', linestyle='--', alpha=0.8)
    plt.axvline(np.mean(hard_probs), color='red', linestyle='--', alpha=0.8)
    
    plt.title(f'Confidence Distribution Shift\n(P-Value: {p_value:.1e})')
    plt.xlabel('Model Confidence in Correct Answer')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- NEW: Save Logic ---
    if args.save_plot_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(args.save_plot_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save figure
        plt.savefig(args.save_plot_path, dpi=300, bbox_inches='tight')
        print(f"\n[SUCCESS] Graph saved to: {args.save_plot_path}")
    else:
        print("\n[INFO] --save_plot_path not provided. Graph was not saved.")
        plt.show() # Show locally if running in notebook/windowed env

    if args.save_json_path:
        # Create dictionary
        json_output = {
            "metadata": {
                "model": args.model_name,
                "seed_file": args.seed_file,
                "hardened_file": args.hardened_file,
                "max_samples": args.max_samples
            },
            "statistics": {
                "mean_seed_confidence": float(mean_seed),
                "mean_hardened_confidence": float(mean_hard),
                "p_value": float(p_value)
            },
            "data": {
                "seed_confidences": seed_probs,
                "hardened_confidences": hard_probs
            }
        }
    # Ensure directory exists
    # directory = os.path.dirname(args.save_json_path)
    # if directory and not os.path.exists(directory):
    #     os.makedirs(directory)
        
    with open(args.save_json_path, 'w') as f:
        json.dump(json_output, f, indent=4)
    print(f"\n[SUCCESS] Raw results saved to: {args.save_json_path}")

if __name__ == "__main__":
    main()