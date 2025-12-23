from init import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dataset_gen import *
import torch
import argparse

parser = argparse.ArgumentParser(description="Evaluate LLMs on MATH-Perturb")
parser.add_argument("--model_path", type=str, required=True, help="Hugging Face model path")
parser.add_argument("--data_dir", type=str, default="qwedsacf/competition_math", help="Dataset path")

args = parser.parse_args()

PROPOSER_MODEL_NAME = args.model_path # "Qwen/Qwen3-30B-A3B-Instruct-2507"
ORACLE_MODEL_NAME = "gpt-5.2-2025-12-11"
# "gpt-5-2025-08-07" #"gpt-4.1"
# DATASET_NAME = "qwedsacf/competition_math"
DATASET_NAME = args.data_dir
# "openai/gsm8k"# "yentinglin/aime_2025"

DATASET_CONFIG_MAP = {
    "qwedsacf/competition_math": 'default',
    'openai/gsm8k': 'main',
    'yentinglin/aime_2025': 'default'
}
DATASET_CONFIG = DATASET_CONFIG_MAP[DATASET_NAME]
# "main" # "default"

# Output file will be saved in the Kaggle working directory
OUTPUT_FILE = "qwen3_gpt5_MATH_dataset_2.jsonl"

# --- Main Execution Logic ---
pipeline = IterativeHardeningPipeline(
    proposer_model_name=PROPOSER_MODEL_NAME,
    oracle_model_name=ORACLE_MODEL_NAME,
    max_iterations=10, # Try to harden each problem up to 3 times,
    dataset = 'MATH'
)

successful_examples = 0
NUM_SAMPLES_TO_PROCESS = 25 # Start with a small number to test the full pipeline
start_idx = 25
# Load the specified slice of the training data
dataset_slice = f"train[{start_idx}:{NUM_SAMPLES_TO_PROCESS + start_idx}]"
source_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=dataset_slice)

# i=0
with open(OUTPUT_FILE, 'w') as f:
    # Use the notebook-friendly tqdm for the progress bar
    for problem in tqdm(source_dataset, desc="Generating Adversarial Data"):
        print(problem)
        # if i < start_idx:
        #     print("continuing", i)
        #     i+=1
        #     continue
        adv_example = pipeline.process_problem(problem)
        if adv_example:
            print("success", adv_example)
            f.write(json.dumps(adv_example) + '\n')
            successful_examples += 1
        # i+=1
print(f"PIPELINE COMPLETE!")
print(f"Successfully generated {successful_examples} adversarial preference pairs.")
print(f"Output saved to: {OUTPUT_FILE}")