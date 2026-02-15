import argparse
import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
# REMOVED: from unsloth import FastLanguageModel
from vllm import LLM, SamplingParams # ADDED
from transformers import AutoTokenizer
from math_verify import parse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
math_perturb_dir = os.path.join(current_dir, 'eval', "MATH_Perturb")
print("MATH perturb dir: " + math_perturb_dir)

if math_perturb_dir not in sys.path:
    sys.path.append(math_perturb_dir)
from eval.MATH_Perturb.evaluate_perturb import extract_ground_truth_answer

# vLLM handles batching internally and very efficiently
BATCH_SIZE = 64

# --- 1. Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Validation Pipeline with Optional Plot Saving")
    parser.add_argument("--seed_file", type=str, required=True, help="Path to original/seed dataset JSON/JSONL")
    parser.add_argument("--hardened_file", type=str, required=True, help="Path to hardened adversarial dataset JSON/JSONL")
    # Note: vLLM usually works better with non-bnb-4bit models unless using AWQ/GPTQ
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct") 
    parser.add_argument("--max_samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--save_plot_path", type=str, default=None, 
                        help="Filename to save the graph. If not provided, graph is not saved.")
    parser.add_argument("--save_json_path", type=str, default=None, help="Path to save raw results for external graphing")

    return parser.parse_args()

# --- 2. Data Loader (Identical to original) ---
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
    
    valid_data = []
    for item in data:
        if seed:
            p = item.get('prompt') or item.get('question') or item.get('problem')
            a = item.get('generated_response') if item.get('is_correct') == True else ""
            # if item.get('is_correct') == True else item.get('ground_truth') or item.get('answer') or item.get('solution') or item.get('original_GT')
        else:
            p = item.get('prompt')
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

# --- MODIFIED: FAST BATCHED INFERENCE USING VLLM ---
def get_answer_probe_probs(llm, dataset, tokenizer):
    probs = []
    trigger_text = "\n\nTherefore, final answer:"
    
    formatted_prompts = []
    target_tokens_list = []

    print(f"Preparing {len(dataset)} samples for vLLM...")
    
    for item in dataset:
        prompt = item['p']
        answer = str(item['a']).strip()
        extracted_answer = extract_answer(answer)
        
        # Robust string cleaning (Identical to original)
        delimiter = "Output format:\n-Steps: (multiple lines)\n-Final answer: \\boxed{...}  (last line only)\n\n"
        if delimiter in prompt:
            output_str = prompt.split(delimiter)[1]
        else:
            output_str = prompt
        extracted_problem = output_str.strip()

        # Construct the "Forced" context
        # We use the tokenizer associated with the LLM object to format chat
        full_text = f"System: {training_prompt}\nUser: {extracted_problem} {trigger_text} {extracted_answer}"
        # formatted_prompts.append(full_text)

        token_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": training_prompt},
                {"role": "user", "content": extracted_problem + trigger_text},
                {"role": "assistant", "content": extracted_answer}
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted_prompts.append({"prompt_token_ids": token_ids})
        
    # vLLM SamplingParams: 
    # prompt_logprobs=1 returns the logprobs of tokens already in the prompt
    sampling_params = SamplingParams(
        max_tokens=1, 
        prompt_logprobs=1, 
        temperature=0.7,
        top_p=0.8,  
        top_k=20,
        stop=[tokenizer.eos_token]
    )

    # Execute Batch Inference
    outputs = llm.generate(formatted_prompts, sampling_params)

    for output in outputs:
        # prompt_logprobs is a list of dicts for each token in the prompt
        # We want the logprob of the VERY LAST token (the answer token)
        # Note: Index 0 is often None, so we look at the last element
        if output.prompt_logprobs and len(output.prompt_logprobs) > 0:
            # Get the logprob dictionary for the last token in the prompt
            last_token_logprobs = output.prompt_logprobs[-1]
            
            # Find the actual token ID that was present at that position
            # vLLM's prompt_logprobs[i] contains the logprob of the token at index i
            # given tokens 0 to i-1.
            # We want the logprob of the answer token that was actually provided.
            actual_token_id = output.prompt_token_ids[-1]
            
            if actual_token_id in last_token_logprobs:
                log_p = last_token_logprobs[actual_token_id].logprob
                probs.append(np.exp(log_p))
            else:
                # Fallback if for some reason the token ID isn't in top logprobs
                probs.append(0.0)
        else:
            probs.append(0.0)

    return probs


def get_batch_sequence_confidence(llm, dataset, tokenizer):
    """
    Calculates the geometric mean of token probabilities for the answer part
    using vLLM's batched prompt_logprobs.
    """
    # tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    prompt_lengths = []
    
    print(f"Tokenizing {len(dataset)} samples...")
    for item in dataset:
        prompt = item['p']
        answer = str(item['a']).strip()
        # print("Prompt: " + prompt)
        # print("Answer: " + answer)
        if answer != "":
            extracted_answer = extract_answer(answer)
            trigger_text = "\n\nTherefore, final answer:"
            
            # Robust string cleaning (Identical to original)
            delimiter = "Output format:\n-Steps: (multiple lines)\n-Final answer: \\boxed{...}  (last line only)\n\n"
            if delimiter in prompt:
                output_str = prompt.split(delimiter)[1]
            else:
                output_str = prompt
            extracted_problem = output_str.strip()

            # Construct the "Forced" context
            # We use the tokenizer associated with the LLM object to format chat
            full_text = f"System: {training_prompt}\nUser: {extracted_problem} {trigger_text} {extracted_answer}"
            # formatted_prompts.append(full_text)

            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": training_prompt},
                    {"role": "user", "content": extracted_problem },
                    # {"role": "assistant", "content": extracted_answer}
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # formatted_prompts.append({"prompt_token_ids": token_ids})
            
            # Calculate how many tokens are in the prompt
            # (This is needed to isolate the answer tokens later)
            prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_token_ids)
            
            # 2. Construct Full Sequence: Prompt + Answer
            full_text = prompt_text + answer
            
            formatted_prompts.append(full_text)
            prompt_lengths.append(prompt_len)

    # 3. vLLM Batch Inference
    # We set prompt_logprobs=1 to get the logprobs of the tokens we provided in the prompt.
    # max_tokens=1 because we don't need the model to generate anything new.
    sampling_params = SamplingParams(
        max_tokens=4096, 
        prompt_logprobs=1, 
        temperature=0
    )

    print(f"Running vLLM inference on {len(dataset)} samples...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    confidences = []
    for i, output in enumerate(outputs):
        # prompt_logprobs is a list of dicts: [ {token_id: LogprobObj}, ... ]
        # It corresponds to the input sequence. 
        logprobs_list = output.prompt_logprobs
        token_ids = output.prompt_token_ids
        
        # The answer starts at the index where the prompt ended
        start_idx = prompt_lengths[i]
        
        answer_logprobs = []
        # Iterate through the tokens belonging to the answer
        for j in range(start_idx, len(token_ids)):
            if logprobs_list[j] is not None and token_ids[j] in logprobs_list[j]:
                lp = logprobs_list[j][token_ids[j]].logprob
                answer_logprobs.append(lp)
        
        if len(answer_logprobs) > 0:
            # Geometric Mean: exp(mean(log_probs))
            mean_nll = np.mean(answer_logprobs)
            print("length mean: ", len(answer_logprobs))
            confidences.append(np.exp(mean_nll))
        else:
            confidences.append(0.0)
            print("something not right")
            print(answer)
            
    return confidences

# --- 4. Main Execution ---
def main():
    args = parse_args()
    
    # Load Model with vLLM
    print(f"Loading vLLM Model: {args.model_name}")
    # vLLM handles multi-GPU and memory automatically. 
    # gpu_memory_utilization can be adjusted if you run into OOM.
    llm = LLM(
        model=args.model_name,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Load Data
    seed_data = load_data(args.seed_file, args.max_samples, seed=True)
    hard_data = load_data(args.hardened_file, args.max_samples, seed=False)

    
    
    # Get Probabilities via vLLM
    seed_probs = get_batch_sequence_confidence(llm, seed_data, tokenizer)
    hard_probs = get_batch_sequence_confidence(llm, hard_data, tokenizer)

    mean_seed = np.mean(seed_probs)
    mean_hard = np.mean(hard_probs)
    
    if not seed_probs or not hard_probs:
        print("Error: Could not extract probabilities. Check dataset format.")
        return

    # Statistics (Identical to original)
    u_stat, p_value = stats.mannwhitneyu(seed_probs, hard_probs, alternative='greater')
    print(f"\nMean Confidence (Seed): {np.mean(seed_probs):.4f}")
    print(f"Mean Confidence (Hardened): {np.mean(hard_probs):.4f}")
    print(f"P-Value: {p_value:.5e}")

    # --- 5. Visualization & Saving (Identical to original) ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(seed_probs, fill=True, label='Original Dataset', color='blue', alpha=0.3, clip=(0,1))
    sns.kdeplot(hard_probs, fill=True, label='Hardened Dataset', color='red', alpha=0.3, clip=(0,1))
    plt.axvline(np.mean(seed_probs), color='blue', linestyle='--', alpha=0.8)
    plt.axvline(np.mean(hard_probs), color='red', linestyle='--', alpha=0.8)
    plt.title(f'Confidence Distribution Shift\n(P-Value: {p_value:.1e})')
    plt.xlabel('Model Confidence in Correct Answer')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if args.save_plot_path:
        directory = os.path.dirname(args.save_plot_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(args.save_plot_path, dpi=300, bbox_inches='tight')
        print(f"\n[SUCCESS] Graph saved to: {args.save_plot_path}")
    else:
        print("\n[INFO] --save_plot_path not provided. Graph was not saved.")
        plt.show()

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