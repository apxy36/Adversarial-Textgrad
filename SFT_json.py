import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Adjust as needed
from unsloth import FastLanguageModel
import json
import re
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from pathlib import Path
from math_verify import parse, verify
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

# ==========================================
# 1. HELPER FUNCTIONS (Answer Extraction)
# ==========================================
# Adapted from your provided evaluation script logic

def extract_answer_value(text: str) -> Optional[str]:
    """
    Extracts the numerical answer from a solution trace.
    Prioritizes \boxed{}, then "The answer is", then the last number.
    """
    if not text: return None
    
    # 1. Check for \boxed{123} (LaTeX style - common in AIME/Math)
    # Priority 1: Search for all \boxed{...} LaTeX commands.
    try:
        boxed_matches = re.findall(r'\\boxed\{([^{}]+)\}', text)
        if boxed_matches:
            return boxed_matches[-1].strip()
    except:
        pass
    
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

    # 3. Fallback: Last number in the text
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    
    return None

def normalize_answer(answer: str) -> Optional[float]:
    """Normalizes string answers to floats for comparison."""
    try:
        if answer is None: return None
        clean_ans = answer.replace(",", "").replace("$", "").replace(' ', '').strip()
        if clean_ans.endswith('.'): clean_ans = clean_ans[:-1]
        return float(clean_ans)
    except (ValueError, TypeError):
        return None

def check_correctness(generated_output, ground_truth):
    """
    Compares generated answer against ground truth.
    Returns: (bool: is_correct, str: extracted_answer)
    """
    # Extract answer from generation
    extracted_gen = extract_answer_value(generated_output)
    
    # Clean up ground truth (handle GSM8K '####' format if present)
    ground_truth_str = str(ground_truth)
    if "####" in ground_truth_str:
        ground_truth_str = ground_truth_str.split("####")[-1].strip()
    
    # Attempt numeric comparison
    norm_gen = normalize_answer(extracted_gen)
    norm_gt = normalize_answer(ground_truth_str)
    
    if norm_gen is not None and norm_gt is not None:
        # Floating point comparison with tolerance
        return abs(norm_gen - norm_gt) < 1e-4, extracted_gen
    
    # Fallback to exact string match
    return str(extracted_gen).strip() == ground_truth_str.strip(), extracted_gen


def check_MATH500(generated_output, ground_truth):
    answer_index = generated_output.find("The answer is: ")

    # check for nonetype object
    if answer_index != -1:
        extracted_gen = generated_output[answer_index + len("The answer is: "):].strip()
        # extracted_gen = extracted_gen.replace(",", "").replace("$", "")
        # extracted_gen = normalize_answer(extracted_gen[0]) if extracted_gen else None
    else:
        extracted_gen = extract_answer_value(generated_output)

    # ans_correct = verify(ground_truth, extracted_gen)
    # ans_correct = answer_check('', extracted_gen, ground_truth, 'original')
    gold = parse(f"${ground_truth}$")
    answer = parse(generated_output)
    try:
        ans_correct = verify(gold, answer)
    except:
        return False, extracted_gen
    
    if not ans_correct:
        try:
            ans_correct = answer_check('', generated_output, ground_truth, 'original')
        except:
            return False, extracted_gen
    return ans_correct, answer
# ==========================================
# 2. CUSTOM CALLBACK FOR EVALUATION
# ==========================================
class MathEvalCallback(TrainerCallback):
    """
    A custom callback to run generative evaluation on the validation set 
    at regular checkpoints.
    """
    def __init__(self, eval_dataset, tokenizer, max_seq_length, output_dir, num_samples=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "eval_history.jsonl")
        # If num_samples is set, we only evaluate on a subset to save time
        self.eval_subset = eval_dataset.select(range(min(len(eval_dataset), num_samples))) if num_samples else eval_dataset

    def on_evaluate(self, args, state, control, model, **kwargs):
        print(f"\n[Epoch {state.epoch:.2f} | Step {state.global_step}] Running Math Validation...")
        
        step = state.global_step
        # 1. Switch to Inference Mode (Unsloth specific optimization)
        FastLanguageModel.for_inference(model)
        
        correct_count = 0
        total_count = 0
        
        # 2. Iterate through validation dataset
        # We wrap in tqdm for a progress bar
        for example in tqdm(self.eval_subset, desc="Evaluating"):
            prompt = example['problem'] # problem
            ground_truth = example['solution']
            
            # Format message for Chat Model
            messages = [
                {"role": "system", "content": 
                 """Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                Solve the following problem.
                - Provide a direct final step-by-step solution in a single pass.
                - Put your final answer on its own line as \\boxed{{...}}.
                - Use exactly one \\boxed{{}} and do not box intermediate results.
                
                Output format:
                -Steps: (multiple lines)
                -Final answer: \\boxed{{...}}  (last line only)"""                
                 },
                {"role": "user", "content": prompt}
            ]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
                enable_thinking = False, 
            ).to("cuda")

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=4096, # Enough for reasoning
                    use_cache=True,
                    temperature=0.7,    # Greedy decoding for eval
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p = 0.8, # for Qwen
                    top_k = 20, 
                    min_p = 0.0,
                )
            
            # Decode
            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract Assistant Response (Strip user prompt)
            # Depending on template, it might contain the prompt. 
            # Quick split hack to get the last part if 'assistant' is in text:
            if "assistant" in generated_text:
                response_text = generated_text.split("assistant")[-1].strip()
            else:
                # If chat template removed tags, we might need to strip prompt length
                # Simpler: just pass the whole text to extractor, it usually finds the last number/box
                response_text = generated_text

            # Check Answer
            is_correct, extracted = check_MATH500(response_text, ground_truth)

            
            
            
            if is_correct:
                correct_count += 1
                print("CORRECT")
            else:
                print("INCORRECT")
                print("RESPONSE TEXT: ", response_text[-250:])
                print("GROUND TRUTH: ", ground_truth[-250:])
            total_count += 1

        # 3. Calculate Metrics
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"\n>>> Validation Results - Step {state.global_step}")
        print(f">>> Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        print("---------------------------------------------------\n")

        # Log to file
        log_entry = {"step": step, "accuracy": accuracy, "correct": correct_count, "total": total_count}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        # Log to wandb or trainer history if needed
        # (transformers logger usually handles printed logs)

        # 4. Switch back to Training Mode
        FastLanguageModel.for_training(model)


# ==========================================
# 3. MAIN SCRIPT
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSON")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation JSON")
    parser.add_argument("--output_dir", type=str, default="qwen_math_output")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_subset", type=int, default=None, help="Limit number of training examples")
    parser.add_argument("--eval_subset", type=int, default=50, help="Number of examples to use for validation check")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--eval_every_steps", type=int, default=50)
    parser.add_argument("--save_limit", type=int, default=None, 
                        help="Max number of checkpoints to keep on disk. Default is None (Save All Checkpoints).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--max_train_epochs", type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()

    # --- Load Model ---
    MAX_SEQ_LENGTH = 32768 # Adjust based on GPU memory
    print(f"Loading {args.model_name}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = False,
        device_map = {"": 0} 
    )

    # --- LoRA Config ---
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

    training_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

                Instruction:
                Solve the following problem.
                - Provide a direct final step-by-step solution in a single pass.
                - Put your final answer on its own line as \\boxed{{...}}.
                - Use exactly one \\boxed{{}} and do not box intermediate results.
                
                Output format:
                -Steps: (multiple lines)
                -Final answer: \\boxed{{...}}  (last line only)"""

    # --- Prepare Data ---
    def formatting_prompts_func(examples, problem_field="prompt", solution_field="solution"):
        prompts = examples[problem_field] # problem
        responses = examples[solution_field]
        texts = []
        for p, r in zip(prompts, responses):
            messages = [
                {"role": "user", "content": str(p)},
                {"role": "assistant", "content": '<think> </think>' + str(r)}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking = False)
            texts.append(text)
        return { "text" : texts }

    def formatting_prompts_func2(examples, problem_field="problem", solution_field="solution"):
        prompts = examples[problem_field] # problem
        responses = examples[solution_field]
        texts = []
        for p, r in zip(prompts, responses):
            messages = [
                {"role": "user", "content": str(p)},
                {"role": "assistant", "content": '<think> </think>' + str(r)}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking = False)
            texts.append(text)
        return { "text" : texts }

    # Load Training Data
    print(f"Loading training data: {args.train_file}")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    if args.train_subset:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.train_subset)))
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    # Load Evaluation Data
    print(f"Loading evaluation data: {args.eval_file}")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train") # Split is usually 'train' for json loader

    # 2. Format Eval Data for the Trainer (Calculates Loss)
    formatted_eval_dataset = eval_dataset.map(formatting_prompts_func2, batched=True)
    # --- Trainer Setup ---
    
    # Initialize our custom callback
    # We pass the raw eval_dataset (before formatting) because the callback 
    # performs generation from the prompt, not training on tokens.
    eval_callback = MathEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        num_samples=args.eval_subset, # Validate on X examples to save time
        output_dir=args.output_dir
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = formatted_eval_dataset, 
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            output_dir = args.output_dir,
            per_device_train_batch_size = args.per_device_train_batch_size if hasattr(args, 'per_device_train_batch_size') else 4,
            gradient_accumulation_steps = 1,
            warmup_ratio = 0.03,
            # max_steps = args.max_steps,
            num_train_epochs = 5,
            learning_rate = args.learning_rate, # 5e-6
            fp16 = False,
            bf16 = True, # True
            # fp8 = True,
            optim = "adamw_torch",
            logging_steps = 10,
            
            # --- Saving & Evaluation Strategy ---
            eval_strategy = "epoch",
            # eval_steps = args.eval_every_steps, 
            save_strategy = "epoch",
            # save_steps = args.eval_every_steps,
        
            
            # Controls how many checkpoints are kept
            # Default None = Keep All. 
            save_total_limit = args.save_limit, 
            
            seed = 3407,
        ),
        # Add the custom callback here
        callbacks=[eval_callback],

    )

    # --- Run ---
    print("Starting Training...")
    trainer.train()

    print(f"Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()