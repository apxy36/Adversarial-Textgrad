import os
import sys
import json
from tqdm import tqdm

# --- 1. SYSTEM CONFIGURATION (Must be at top) ---
# Fixes for multi-GPU stability based on your previous errors
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Adjust based on your GPUs

# Import vLLM after env vars
from vllm import LLM, SamplingParams
from datasets import load_dataset

class BatchVLLMAgent:
    def __init__(self, model_path: str):
        print(f">>> Initializing vLLM Agent with model: {model_path}")
        
        # Initialize vLLM with the stability flags we discovered
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=2, # Number of GPUs
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            
            # CRITICAL STABILITY FLAGS
            disable_custom_all_reduce=True, 
            enforce_eager=True,             
        )
        
        self.sampling_params = SamplingParams(
            temperature=0,      # Greedy decoding for evaluation (standard for MATH)
            top_p=0.9,
            max_tokens=32768,    # Long enough for chain-of-thought
            stop=["<|eot_id|>", "<|end_of_text|>", "Problem:"] 
        )

    def generate_batch(self, problems: list[str]) -> list[str]:
        """
        Takes a list of problem strings and returns a list of generated solutions.
        """
        # 1. Format prompts (Apply Chat Template)
        # We manually format here to ensure efficient list processing
        tokenizer = self.llm.get_tokenizer()
        prompts = []
        
        for problem in problems:
            # Standard MATH prompt template
            content = (
                f"Below is an instruction that describes a task. "
                f"Write a rigorous and appropriate step-by-step solution. "
                f"Put your final answer within \\boxed{{}}.\n\n"
                f"### Instruction:\n{problem}\n\n### Response:"
            )
            prompts.append(content)

        # 2. Batch Inference
        # vLLM handles the internal batching logic automatically.
        # We just pass the whole list.
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)

        # 3. Extract Text
        results = [output.outputs[0].text for output in outputs]
        return results

def eval_math500():
    # --- CONFIGURATION ---
    MODEL_PATH = "unsloth/llama-3-8b-bnb-4bit" # Replace with your model
    OUTPUT_FILE = "math500_results.jsonl"
    BATCH_SIZE = 64  # Process 64 questions at a time (Adjust based on VRAM)
    
    # 1. Load Data
    print(">>> Loading MATH-500 Dataset...")
    try:
        # Try loading from HuggingFace
        dataset = load_dataset("HuggingFaceH4/math-500", split="test")
    except:
        print("Could not load from HF. Ensure 'datasets' is installed or load local json.")
        return

    problems = dataset['problem']
    solutions = dataset['solution'] # Ground truth (for checking later)
    
    # 2. Initialize Agent
    agent = BatchVLLMAgent(MODEL_PATH)
    
    # 3. Batch Evaluation Loop
    total_samples = len(problems)
    print(f">>> Starting Inference on {total_samples} samples with Batch Size {BATCH_SIZE}...")
    
    # We process in chunks so we can save results incrementally
    # (If the script crashes at item 499, we don't want to lose the first 498)
    all_results = []
    
    # Clear output file
    with open(OUTPUT_FILE, 'w') as f:
        pass

    for i in range(0, total_samples, BATCH_SIZE):
        # Slice the batch
        batch_problems = problems[i : i + BATCH_SIZE]
        batch_indices = range(i, min(i + BATCH_SIZE, total_samples))
        
        print(f"Processing batch {i} to {i + len(batch_problems)}...")
        
        # RUN INFERENCE
        batch_generated = agent.generate_batch(batch_problems)
        
        # Save Results
        with open(OUTPUT_FILE, 'a') as f:
            for idx, gen_sol in zip(batch_indices, batch_generated):
                result_entry = {
                    "id": idx,
                    "problem": problems[idx],
                    "ground_truth": solutions[idx],
                    "generated_solution": gen_sol
                }
                f.write(json.dumps(result_entry) + "\n")
                all_results.append(result_entry)

    print(f">>> Evaluation Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    eval_math500()