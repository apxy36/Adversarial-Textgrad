from init import *
from dataset_gen import *
import torch

# Cell 4: Configuration and Execution

PROPOSER_MODEL_NAME = "Qwen/Qwen3-4B" # "Qwen/Qwen3-30B-A3B-Instruct-2507"
ORACLE_MODEL_NAME = "gpt-5-2025-08-07" #"gpt-4.1"
DATASET_NAME = "openai/gsm8k"# "yentinglin/aime_2025"
DATASET_CONFIG = "main" # "default"

# Output file will be saved in the Kaggle working directory
OUTPUT_FILE = "qwen3_gpt5_dpo_dataset.jsonl"

# --- Main Execution Logic ---
pipeline = IterativeHardeningPipeline(
    proposer_model_name=PROPOSER_MODEL_NAME,
    oracle_model_name=ORACLE_MODEL_NAME,
    max_iterations=8 # Try to harden each problem up to 3 times
)

successful_examples = 0
NUM_SAMPLES_TO_PROCESS = 1 # Start with a small number to test the full pipeline
# Load the specified slice of the training data
dataset_slice = f"train[:{NUM_SAMPLES_TO_PROCESS}]"
source_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=dataset_slice)
start_idx = 0
i=0
with open(OUTPUT_FILE, 'w') as f:
    # Use the notebook-friendly tqdm for the progress bar
    for problem in tqdm(source_dataset, desc="Generating Adversarial Data"):
        print(problem)
        if i < start_idx:
            print("continuing", i)
            i+=1
            continue
        adv_example = pipeline.process_problem(problem)
        if adv_example:
            print("success", adv_example)
            f.write(json.dumps(adv_example) + '\n')
            successful_examples += 1
        i+=1
print(f"PIPELINE COMPLETE!")
print(f"Successfully generated {successful_examples} adversarial preference pairs.")
print(f"Output saved to: {OUTPUT_FILE}")