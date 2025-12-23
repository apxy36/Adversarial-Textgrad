import sys
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"




# from  import UnslothAgent
# import agents from parent directory
from pathlib import Path
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.append(str(parent_dir))

from agents.unslothagent import UnslothAgent
# 1. Get the path to the 'MATH_Perturb' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
math_perturb_dir = os.path.join(current_dir, 'MATH_Perturb')

# 2. Add it to Python's search path
if math_perturb_dir not in sys.path:
    sys.path.append(math_perturb_dir)
import json
# Now try the import again
from MATH_Perturb.evaluate_perturb import answer_check
from unsloth import FastLanguageModel
import re
import argparse
import glob
from tqdm import tqdm
from sympy.parsing.latex import parse_latex
from sympy import simplify
import torch
# from evaluate_perturb import answer_check


# ==========================================
# 1. Robust File Discovery
# ==========================================

def find_dataset_file(data_dir, keywords):
    """
    Searches for a .jsonl file in data_dir that contains all keywords (case-insensitive).
    """
    if not os.path.exists(data_dir):
        return None
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    
    for f in files:
        if all(k.lower() in f.lower() for k in keywords):
            return f
            
    return None


class UnslothEvaluator:
    def __init__(self, model_path, max_seq_length=32768, load_in_4bit=True):
        print(f"Loading Unsloth model: {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)

    def generate_solution(self, problem_text):
        prompt = (
            "Problem:\n"
            f"{problem_text}\n\n"
            
            "Please reason step by step and put your final answer within \\boxed{}. \n"
            "Solution:\n"
        )
        print("Prompt for generation:" + prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=32768, 
                do_sample=False, 
                use_cache=True,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in response:
            solution = response.split(prompt)[-1]
        else:
            solution = response[len(prompt):]

        print(f"Generated Solution: {solution}")
        return solution

# ==========================================
# 4. Main Evaluation Loop
# ==========================================

def evaluate_dataset(dataset_path, model_evaluator, output_file, limit=None):
    print(f"Evaluating dataset: {dataset_path}")
    
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if limit is not None and limit > 0:
        print(f"Limiting evaluation to first {limit} examples.")
        data = data[1:limit+1]

    results = []
    correct_count = 0

    for item in tqdm(data):
        problem = item['problem']
        ground_truth = item['answer']
        
        prediction = model_evaluator.solve(problem)
        # prediction = "\\boxed{42}"  # Placeholder for testing
        is_correct = answer_check('', prediction, ground_truth, 'perturb')
        
        if is_correct:
            correct_count += 1
            
        results.append({
            "problem_id": item.get('problem_id'),
            "problem": problem,
            "ground_truth": ground_truth,
            "model_prediction": prediction,
            "is_correct": is_correct
        })

    accuracy = correct_count / len(data) if data else 0
    print(f"Accuracy on {os.path.basename(dataset_path)}: {accuracy:.2%}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {output_file}")

# ==========================================
# 5. Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLMs on MATH-Perturb")
    parser.add_argument("--model_path", type=str, required=True, help="Hugging Face model path")
    parser.add_argument("--data_dir", type=str, default="math_perturb", help="Directory containing jsonl files")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to evaluate per dataset")
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--no_4bit", action="store_true")
    
    args = parser.parse_args()

    # debug: Print current working directory and data_dir contents
    print(f"Current Working Directory: {os.getcwd()}")
    if os.path.exists(args.data_dir):
        print(f"Contents of '{args.data_dir}':")
        print(os.listdir(args.data_dir))
    else:
        print(f"ERROR: Data directory '{args.data_dir}' does not exist.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Model
    # evaluator = UnslothEvaluator(
    #     args.model_path, 
    #     max_seq_length=args.max_seq_length,
    #     load_in_4bit=False
    # )

    evaluator_2 = UnslothAgent(
        model_name=args.model_path,
        prompt="You are an expert mathematician. Solve the following problem step-by-step and provide the final answer in \\boxed{}."
    ) # set cuda to 2

    # Auto-discover files based on keywords "simple" and "hard"
    target_datasets = [
        {"keywords": ["simple"], "type": "simple"},
        {"keywords": ["hard"], "type": "hard"}
    ]

    found_files = False
    for target in target_datasets:
        filename = find_dataset_file(args.data_dir, target["keywords"])
        
        if filename:
            found_files = True
            full_path = os.path.join(args.data_dir, filename)
            clean_model_name = os.path.basename(args.model_path)
            output_filename = f"{clean_model_name}_on_{filename.replace('.jsonl', '.json')}"
            output_path = os.path.join(args.output_dir, output_filename)
            
            print(f"Found {target['type']} dataset: {filename}")
            evaluate_dataset(full_path, evaluator_2, output_path, limit=args.limit)
        else:
            print(f"Warning: Could not find a .jsonl file containing keywords {target['keywords']} in {args.data_dir}")

    if not found_files:
        print("\nPossible fix: Check your --data_dir path.")
        print("If you are in 'MATH-Perturb/', try: --data_dir math_perturb")
        print("If you are in the parent dir, try:   --data_dir MATH-Perturb/math_perturb")