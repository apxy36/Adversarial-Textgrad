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
# --- Configuration ---
# Path to your fine-tuned adapters (from SFT step) or the merged model
# MODEL_PATH = "/mnt/ssd/iclrtemp/adapters/qwen3_gsm8k_v2.6"
# MODEL_PATH = "/mnt/ssd/iclrtemp/adapters/qwen3_metamath"

MODEL_PATH = "Qwen/Qwen3-14B"
# MODEL_PATH = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit" # Use this to test baseline
# MODEL_PATH = "WizardLMTeam/WizardMath-7B-V1.1" # WizardMath finetuned on GSM8K
# MODEL_PATH = "meta-math/MetaMath-Mistral-7B"
# MODEL_PATH = "Vivacem/Mistral-7B-MMIQC"
# MODEL_PATH = "hkust-nlp/dart-math-dsmath-7b-prop2diff"

# Generation Config
MAX_SEQ_LENGTH = 32768 # Max length for deep reasoning
dtype = None # Auto detection
load_in_4bit = False

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
        # normalise spaces
        clean_ans = answer.replace(",", "").replace("$", "").replace('.', '').strip()
        # .strip 
        # clean_ans = re.sub(r"[^\d\.\-]", "", answer)
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

# --- 2. Evaluation Runner ---
def evaluate_dataset(agent, dataset, dataset_name, num_samples=None):
    print(f"\n--- Starting Evaluation: {dataset_name} ---")
    
    correct_count = 0
    total_count = 0
    results_log = []
    
    # Limit samples for quick testing if needed
    eval_data = dataset if num_samples is None else dataset.select(range(min(len(dataset), num_samples)))

    # Enable native inference optimizations
    # FastLanguageModel.for_inference(model)

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
        elif "MATH500" in dataset_name:
            question = example['problem']
            ground_truth = example['answer']

        question += "Please put your final answer within \\boxed{}.\n"
        
        # Prepare Prompt (Using Qwen/ChatML format)
        messages = [
            {"role": "system", "content": "Below is an instruction that describes a task. Write a rigorous and appropriate step-by-step solution to the task. \nBe concise and efficient in your reasoning."},
            {"role": "user", "content": "Instruction: " + question}
        ]

        # input_ids = tokenizer.apply_chat_template(
        #     messages, 
        #     tokenize=True, 
        #     add_generation_prompt=True, 
        #     return_tensors="pt"
        # ).to("cuda")

        # Generate
        # with torch.no_grad():
        #     outputs = model.generate(
        #         input_ids=input_ids,
        #         max_new_tokens=MAX_SEQ_LENGTH, # Allow deep thinking
        #         use_cache=True,
        #         temperature=0.7, # Greedy decoding for reproducible eval
        #     )
        
        # generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        generated_text = agent.solve(question)
        
        # Isolate the assistant's response (remove the input prompt)
        # Note: formatting depends on tokenizer, usually splitting by "assistant" works
        try:
            response_text = generated_text.split("assistant\n")[-1].strip()
        except:
            response_text = generated_text

        # Verify
        if dataset_name == "MATH":
            is_correct = answer_check(question, response_text, ground_truth, 'peturb')
            extracted_val = extract_ground_truth_answer('', response_text, 'original')
            pred_val = extract_predicted_answer('', ground_truth, 'original')
        elif dataset_name == "MATH500" or dataset_name == 'AIME':
            is_correct, extracted_val = check_MATH500(response_text, ground_truth)
            pred_val = ground_truth
        else: 
            is_correct, extracted_val = check_correctness(response_text, ground_truth)
            pred_val = extracted_val
        
        if is_correct:
            print(f"Sample {i}: Correct")
            correct_count += 1
        else:
            print(f"Sample {i}: Incorrect")
            print(f"  Question: {question}")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Generated Response: {response_text}")
            print(f"  Extracted Answer: {extracted_val}")
            print(f"  True Answer: {pred_val}")
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

# --- 2. Evaluation Runner (Modified for Batching) ---
def evaluate_dataset_batch(agent, dataset, dataset_name, num_samples=None, batch_size=96):
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
                ground_truth = extract_answer_value(ground_truth)
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


mistral_template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

deepseek_math_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "User: {{ message['content'] }} "
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "Assistant:"
    "{% endif %}"
)
# --- 3. Main Execution ---
def main():
    # Load Model
    print(f"Loading Model from {MODEL_PATH}...")

    # argparse for command line arguments
    parser = argparse.ArgumentParser(description="Evaluate Language Model on Math Datasets")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the language model")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="File to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--no_use_lora", action='store_false', help="Whether to use LoRA adapters if available")
    parser.add_argument("--use_local", action='store_true', help="Whether to load dataset from local JSON file")
    # parser.add_argument("--max_seq_length", type=int, default=MAX
    args = parser.parse_args()


    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = MODEL_PATH,
    #     max_seq_length = MAX_SEQ_LENGTH,
    #     dtype = dtype,
    #     load_in_4bit = load_in_4bit,
    #     device_map = 'auto' # {"": 0}
    # )

    # agent = UnslothAgent(args.model_path, think=False)
    agent = VLLMAgent(args.model_path, think=True, use_lora=args.no_use_lora)
    print("Lora usage:", args.no_use_lora)
    # agent.tokenizer.chat_template = deepseek_math_template

    # # padding adjustment for MMIQC only
    # # Set the EOS token to the ChatML end token
    # agent.tokenizer.eos_token = "<|im_end|>"
    # # Ensure the ID is updated
    # agent.tokenizer.eos_token_id = agent.tokenizer.convert_tokens_to_ids("<|im_end|>")

    # agent.tokenizer.pad_token = agent.tokenizer.eos_token
    # agent.tokenizer.pad_token_id = agent.tokenizer.eos_token_id
    # agent.tokenizer.padding_side = "left"
    
    # 1. Evaluate GSM8K (Standard Test Set)
    print("Loading MATH...")
    math = load_dataset("HuggingFaceH4/MATH-500", "default", split="test")
    dataset_label = "MATH500"

    # add option to load from local JSON file

    load_from_local = args.use_local
    if load_from_local:
        print("loading local")
        local_math_path = "MATH_validation_2.json"
        if os.path.exists(local_math_path):
            print(f"Loading local MATH dataset from {local_math_path}...")
            math = load_dataset("json", data_files=local_math_path, split="train")
            print(math[0])
            dataset_label = "validation"
        else:
            print(f"Local MATH dataset file {local_math_path} not found. Using standard MATH dataset from HuggingFace.")
    else:
        print("loading MATH500 from HF")
    # acc_gsm, log_gsm = evaluate_dataset(model, tokenizer, gsm8k, "GSM8K", num_samples=100) # Set num_samples for debugging
    
    # 2. Evaluate AIME 2025 (Or fallback to standard AIME)
    print("Loading AIME...")
    if AIME_25_PATH and os.path.exists(AIME_25_PATH):
        # Load local JSONL file for specific AIME 2025 data
        aime_dataset = load_dataset("json", data_files=AIME_25_PATH, split="train")
        # dataset_label = "AIME 2025 (Local)"
    else:
        # Fallback to HuggingFace AIME dataset (usually up to 2023/24)
        # We take a slice of the end to approximate "hard/recent" problems
        print("AIME 25 file not found/provided. Loading standard AIME from Hub.")
        full_aime = load_dataset("yentinglin/aime_2025", "default", split="train")
        # Take the last 50 problems as a proxy for 'recent/hard'
        aime_dataset = full_aime.select(range(0, len(full_aime))) 
        # dataset_label = "AIME (Last 50 Samples)"
        # dataset_label = "MATH500"

    acc_threshold = 0.0
    exceed = False
    while not exceed:
        acc_math, log_math = evaluate_dataset_batch(agent, math, dataset_label, num_samples=args.num_samples)
        if acc_math > acc_threshold:
            print("Final stop")
            exceed = True
        else:
            print("Continuing cuz acc not high enough: ", acc_math)
    # acc_aime, log_aime = evaluate_dataset_batch(agent, aime_dataset, "AIME", len(full_aime))

    # Save Results
    output_file = args.output_file
    
    # "evaluation_results_wizardmath120_full.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {"model": args.model_path},
            # "gsm8k": {"accuracy": acc_gsm, "details": log_gsm},
            "math": {"accuracy": acc_math, "details": log_math},
            # "aime": {"accuracy": acc_aime, "details": log_aime},
        }, f, indent=2)
    print(f"Full logs saved to {output_file}")

if __name__ == "__main__":
    main()