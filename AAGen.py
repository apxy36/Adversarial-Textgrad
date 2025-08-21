from AdvTrain import *
import torch

# Cell 4: Configuration and Execution

# --- Configuration (replaces argparse) ---
PROPOSER_MODEL_NAME = "Qwen/Qwen3-4B" # "Qwen/Qwen3-30B-A3B-Instruct-2507"
ORACLE_MODEL_NAME = "gpt-4.1"
DATASET_NAME = "yentinglin/aime_2025"
DATASET_CONFIG = "default"

# Output file will be saved in the Kaggle working directory
OUTPUT_FILE = "/kaggle/working/qwen2_o1_dpo_dataset.jsonl"

# --- Main Execution Logic ---
pipeline = AdversarialPipeline(
    proposer_model_name=PROPOSER_MODEL_NAME,
    oracle_model_name=ORACLE_MODEL_NAME
)

successful_examples = 0
NUM_SAMPLES_TO_PROCESS = 30 # Start with a small number to test the full pipeline
# Load the specified slice of the training data
dataset_slice = f"train[:{NUM_SAMPLES_TO_PROCESS}]"
source_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=dataset_slice)
start_idx = 2
i=0
with open(OUTPUT_FILE, 'w') as f:
    # Use the notebook-friendly tqdm for the progress bar
    for problem in tqdm(source_dataset, desc="Generating Adversarial Data"):
        if i < start_idx:
            print("continuing", i)
            i+=1
            continue
        adv_example = pipeline.generate_single_example(problem)
        if adv_example:
            print("success", adv_example)
            f.write(json.dumps(adv_example) + '\n')
            successful_examples += 1
        i+=1

print(f"PIPELINE COMPLETE!")
print(f"Successfully generated {successful_examples} adversarial preference pairs.")
print(f"Output saved to: {OUTPUT_FILE}")