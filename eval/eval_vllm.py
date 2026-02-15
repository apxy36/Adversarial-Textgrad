import os, sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Force communication via PCIe instead of NVLink/P2P
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# "1,2"
os.environ["VLLM_USE_V1"] = "0"
from pathlib import Path
from math_verify import parse, verify

current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.append(str(parent_dir))

# from agents.unslothagent import UnslothAgent
# 1. Get the path to the 'MATH_Perturb' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
math_perturb_dir = os.path.join(current_dir, 'MATH_Perturb')
if math_perturb_dir not in sys.path:
    sys.path.append(math_perturb_dir)
import json
from MATH_Perturb.evaluate_perturb import answer_check, extract_predicted_answer, extract_ground_truth_answer
from init import *

from agents.vllmagent import VLLMAgent
# import torch
# from trl import SFTTrainer
# from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import re
import json
from tqdm import tqdm
import argparse

MODEL_PATH = "Qwen/Qwen3-14B"

# Generation Config
MAX_SEQ_LENGTH = 32768 # Max length for deep reasoning
dtype = None # Auto detection
load_in_4bit = False

AIME_25_PATH = None 

def evaluate_dataset_batch(agent, dataset, dataset_name, num_samples=None, batch_size=100):
    print(f"\n--- Starting Evaluation: {dataset_name} | Batch Size: {batch_size} ---")
    
    correct_count = 0
    total_count = 0
    results_log = []
    
    # Slice dataset if num_samples is specified
    if num_samples is not None:
        # Create a subset range
        eval_indices = range(min(len(dataset), num_samples))
        dataset = dataset.select(eval_indices)

    total_samples = len(dataset)
    
    # Iterate in chunks
    for i in tqdm(range(0, total_samples, batch_size), desc="Batch Processing"):
        # 1. Prepare Batch
        # Using .select or direct slicing depending on dataset type, 
        # but standardized HF datasets allow slicing returning dict of lists
        batch_end = min(i + batch_size, total_samples)
        batch_data = dataset[i : batch_end] 
        
        # Convert dict of lists to list of dicts for easier handling if needed,
        # or just iterate the indices relative to the batch.
        # Since 'batch_data' is usually {'col': [val1, val2]}, let's zip them.
        
        # We need to construct a list of prompts for the agent
        prompts = []
        batch_metadata = [] # To store GT and original question to pair with results later

        # Get the keys to iterate safely
        keys = batch_data.keys()
        # Number of items in this specific batch
        current_batch_len = len(list(batch_data.values())[0])

        for idx in range(current_batch_len):
            # Reconstruct the single example dictionary from the batch slice
            example = {k: batch_data[k][idx] for k in keys}

            # Handle dataset-specific fields
            if dataset_name == "GSM8K":
                question = example['question']
                ground_truth = example['answer']
            elif "AIME" in dataset_name:
                question = example.get('problem', example.get('question'))
                ground_truth = example.get('solution', example.get('answer'))
                # ground_truth = extract_answer_value(ground_truth)
            elif "MATH500" in dataset_name:
                question = example['problem']
                ground_truth = example['answer']
            elif "validation" in dataset_name.lower():
                # print("Example:", example)
                question = example["problem"]
                ground_truth = example['solution']
            else:
                # Fallback
                question = example.get('question', '')
                ground_truth = example.get('answer', '')

            # Add instruction
            instruction_prompt = question + " Please put your final answer within \\boxed{}.\n"
            prompts.append(instruction_prompt)
            
            # Store metadata for verification step
            batch_metadata.append({
                "question": instruction_prompt, 
                "original_q": question,
                "ground_truth": ground_truth
            })

        # 2. Run Batch Inference
        # agent.batch_solve returns a list of strings [response1, response2, ...]
        generated_responses = agent.batch_solve(prompts)
        print(f"Generated {len(generated_responses)} responses for batch starting at index {i}.")

        # 3. Process Results
        for j, response_text in enumerate(generated_responses):
            meta = batch_metadata[j]
            question = meta["question"]
            ground_truth = meta["ground_truth"]
            
            # Post-processing (removing assistant/system tokens if leaked)
            try:
                # Basic cleanup if chat template leaks
                if "assistant\n" in response_text:
                    clean_response = response_text.split("assistant\n")[-1].strip()
                else:
                    clean_response = response_text
            except:
                clean_response = response_text

            # Check Correctness
            if dataset_name == "MATH":
                is_correct = answer_check(question, clean_response, ground_truth, 'peturb')
                extracted_val = extract_ground_truth_answer('', clean_response, 'original')
                # pred_val = extract_predicted_answer('', ground_truth, 'original')
            elif dataset_name == "MATH500" or "validation" in dataset_name.lower():
                is_correct, extracted_val = check_MATH500(clean_response, ground_truth)
            else: 
                is_correct, extracted_val = check_correctness(clean_response, ground_truth)
            
            if is_correct:
                correct_count += 1
                print(f"Sample {i + j}: Correct")
            else:
                print(f"Sample {i + j}: Incorrect")
                print(f"  Question: {meta['original_q']}")
                print(f"  Ground Truth: {ground_truth}")
                print(f"  Generated Response: {clean_response[-100:]}")
                print(f"  Extracted Answer: {extracted_val[-100:] if extracted_val else 'None'}")
            
            total_count += 1
            
            # Log specific failures for debugging (optional: reduce print spam for large batches)
            if not is_correct and total_count % 10 == 0: 
                # Only print occasionally to keep tqdm clean
                pass 

            results_log.append({
                "question": meta["original_q"],
                "ground_truth": str(ground_truth),
                "generated_response": clean_response,
                "extracted_answer": str(extracted_val),
                "is_correct": is_correct
            })

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nResults for {dataset_name}:")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    return accuracy, results_log

def main():
    parser = argparse.ArgumentParser(description="Evaluate Language Model on Math Datasets")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the language model")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file (problem states)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="File to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--no_use_lora", action='store_false', help="Whether to use LoRA adapters if available")
    parser.add_argument("--use_local", action='store_true', help="Whether to load dataset from local JSON file")
    # parser.add_argument("--max_seq_length", type=int, default=MAX
    args = parser.parse_args()
    # agent = UnslothAgent(args.model_path, think=False)
    agent = VLLMAgent(args.model_path, think=True, use_lora=args.no_use_lora)
    print("Lora usage:", args.no_use_lora)

    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        problems = json.load(f)

    # Filter for active problems only
    active_indices = [i for i, p in enumerate(problems) if p.get('is_active', True)]
    prompts = [problems[i]['current_text'] for i in active_indices] 

    outputs = agent.batch_solve(prompts)

    # Map outputs back to problems
    print("Mapping results...")
    for idx, output in zip(active_indices, outputs):
        generated_text = output.strip()
        
        # We append the NEW trace to the history buffer temporarily
        # The workstation script will process this into specific history entries
        if 'pending_trace' not in problems[idx]:
            problems[idx]['pending_trace'] = generated_text

    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(problems, f, indent=2)

if __name__ == "__main__":
    main()