import argparse
import json
import torch
import re
from tqdm import tqdm
import random
import os, sys
from pathlib import Path
from math_verify import parse, verify
from agents.vllmagent import VLLMAgent

import wandb

os.environ["WANDB_PROJECT"] = "adversarial-textgrad"  # Set your WandB project name
os.environ["WANDB_LOG_MODEL"] = "checkpoint"     # Log model checkpoints to wandb (optional)

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.append(str(parent_dir))

# from agents.unslothagent import UnslothAgent
# 1. Get the path to the 'MATH_Perturb' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
math_perturb_dir = os.path.join(current_dir, 'eval', "MATH_Perturb")
print("MATH perturb dir: " + math_perturb_dir)

# 2. Add it to Python's search path
if math_perturb_dir not in sys.path:
    sys.path.append(math_perturb_dir)
# from evalua import 
from eval.MATH_Perturb.evaluate_perturb import answer_check, extract_predicted_answer, extract_ground_truth_answer
# ==========================================
# 0. CONFIGURATION & IMPORTS
# ==========================================


# Define strict types for regex
from typing import Optional, Union
# from unsloth import FastLanguageModel

# --- 1. The Unsloth Agent Class ---


# --- 2. Math Verification Helpers ---

def extract_boxed_answer(text):
    """Extracts content inside \\boxed{...}."""
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if matches:
        return matches[-1]
    return None

def normalize_answer_string(s):
    """Normalizes latex strings for comparison (removes whitespace, standardized fractions)."""
    if s is None: return ""
    s = str(s).strip()
    s = s.replace(" ", "").replace("\n", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace("$", "")
    return s

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

def check_MATH500(generated_output, ground_truth):

    if generated_output is None or generated_output.strip() ==  '' or ground_truth is None or ground_truth.strip() == '':
        return False, ''
    
    
    
    # print("OUTPUT ", generated_output, "GT: ", ground_truth)
    answer_index = generated_output.find("The answer is: ")

    # check for nonetype object
    extracted_gen = None
    if answer_index != -1:
        extracted_gen = generated_output[answer_index + len("The answer is: "):].strip()
        # extracted_gen = extracted_gen.replace(",", "").replace("$", "")
        # extracted_gen = normalize_answer(extracted_gen[0]) if extracted_gen else None
    
    if extracted_gen is None:
        if parse(generated_output) is not None or len(parse(generated_output)) > 0:
            extracted_gen = extract_predicted_answer('', generated_output)
    else:
        extracted_gen = extract_answer_value(generated_output)


    gold = parse(f"${ground_truth}$")
    answer = parse(generated_output)
    try:
        ans_correct = verify(gold, answer)
    except:
        return False, extracted_gen
    
    if not ans_correct:
    # ans_correct = verify(ground_truth, extracted_gen)
        try:
            ans_correct = answer_check('', generated_output, ground_truth, 'original')
        except:
            return False, extracted_gen
    return ans_correct, answer
    norm_gen = normalize_answer(extracted_gen)
    norm_gt = normalize_answer(str(ground_truth))
    
    if norm_gen is not None and norm_gt is not None:
        return str(norm_gen).strip() == str(norm_gt).strip(), norm_gen
    
    return str(extracted_gen).strip() == str(ground_truth).strip(), extracted_gen

# --- 3. Main Pipeline Logic ---

def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth Agent RFT Pipeline")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Model ID")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--input_json", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output_json", type=str, required=True, help="Output dataset path")
    parser.add_argument("--num_samples", type=int, default=4, help="Samples per problem")
    # parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--problem_key", type=str, default="problem")
    parser.add_argument("--gt_key", type=str, default="solution")
    parser.add_argument("--target_count", type=int, default=100, help="Target number of valid RFT examples to collect")
    parser.add_argument("--inference_batch_size", type=int, default=8, help="Batch size for inference")
    return parser.parse_args()

def batch_data(dataset, batch_size):
    """Yields chunks of data from the list."""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

system_prompt = """You are an expert reasoning engine and mathematician.

Your task is to solve the problem provided by the user. You will be provided with a "Target Answer" along with the problem. This Target Answer is the ground truth, and your goal is to generate a valid, step-by-step chain of thought that arrives at this specific answer.

CRITICAL INSTRUCTIONS:
1.  **Forward-Chaining Only:** You must solve the problem using forward reasoning (A -> B -> C). Do not use reverse reasoning (starting from the answer and working backward) unless it is a specific mathematical technique required by the problem type.
2.  **No Meta-Commentary:** Do NOT mention the "Target Answer," "Ground Truth," or the fact that you were provided with the solution in your output. Do not say "We are looking for..." or "Let's check if it matches..."
3.  **Organic Tone:** Write the solution as if you are encountering the problem for the first time and solving it confidently from scratch.
4.  **Logical Consistency:** Do not force the logic. If the reasoning does not naturally lead to the Target Answer, adjust your approach or re-read the question constraints until you find the valid logical path that leads to the Target Answer.
5.  **Formatting:**
    - Provide a direct final step-by-step solution in a single pass.
    - Put your final answer on its own line as \\boxed{{...}}.
    - Use exactly one \\boxed{{}} and do not box intermediate results.

    Output format:
    -Steps: (multiple lines)
    -Final answer: \\boxed{{...}}  (last line only)
"""

def main():
    args = parse_args()
    agent = VLLMAgent(model_path=args.model_name, think=True, use_lora=False, prompt=system_prompt)

    print(f"Loading data from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        source_dataset = json.load(f)

    
    # "You are a reasoning expert. Solve the problem step by step. Put your final answer within \\boxed{}."
    # generate as if to reach the final ans, naturally, solve the question as though you do not know the final answer. 
    final_dataset = []
    
    # Calculate Total Effective Batch Size for VRAM warning
    total_batch_size = args.inference_batch_size * args.num_samples
    print(f"Config: Processing {args.inference_batch_size} questions x {args.num_samples} samples.")
    print(f"Effective GPU Batch Size: {total_batch_size}. Ensure your VRAM can handle this.")

    pbar = tqdm(total=args.target_count, desc="Collecting Valid RFT Examples", unit="samples")
    
    # Iterate over dataset in chunks
    data_iterator = batch_data(source_dataset, args.inference_batch_size)
    iterated_count = 0

    for batch_entries in data_iterator:
        # skip first 1000 entries for testing
        
        # if iterated_count < 1000:
        #     iterated_count += len(batch_entries)
        #     continue
        # iterated_count += len(batch_entries)

        if len(final_dataset) >= args.target_count:
            break

        # Extract questions and GTs for this batch
        questions = []
        ground_truths = []
        
        for entry in batch_entries:
            q = entry.get(args.problem_key)
            gt = entry.get(args.gt_key)
            ans = parse(gt)[0]
            if q and gt:
                for _ in range(args.num_samples):
                    prompt = "Problem: " + str(q) + "\nTarget Answer: " + str(ans) + '\nProvide a complete, natural solution to the problem above.'
                    questions.append(prompt)
                    ground_truths.append(gt)

        if not questions:
            continue

        # Generate! 
        # Returns flat list of size (len(questions) * num_samples)
        # e.g., [Q1_S1, Q1_S2, Q2_S1, Q2_S2...]
        flat_outputs = agent.batch_solve(
            questions=questions,
            # system_prompt=system_prompt,
            # n_samples=args.num_samples,
            # temperature=args.temperature
        )

        # Process results and map back to specific questions
        for i, generated_text in enumerate(flat_outputs):
            if len(final_dataset) >= args.target_count:
                break

            # Math magic to find which question this output belongs to
            # index // num_samples gives the index of the original question in the batch
            question_idx = i // args.num_samples
            
            original_q = questions[question_idx]
            original_gt = ground_truths[question_idx]

            if check_MATH500(generated_text, original_gt):
                final_dataset.append({
                    "problem": original_q,
                    "solution": generated_text,
                    "original_GT": original_gt
                })
                pbar.update(1)
                print(f"Collected {len(final_dataset)}/{args.target_count} valid RFT examples.")

    pbar.close()
    print(f"Saving {len(final_dataset)} samples to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(final_dataset, f, indent=2)

if __name__ == "__main__":
    main()
    wandb.finish()