import pandas as pd
import json
import os, sys
from math_verify import parse, verify


# Point this to your final_preference_pairs.jsonl from the TextGrad pipeline
INPUT_FILE = "SFTpairs6.jsonl" 
OUTPUT_PATH = "data/test_rl_data"

current_dir = os.path.dirname(os.path.abspath(__file__))
math_perturb_dir = os.path.join(current_dir, 'eval', "MATH_Perturb")
print("MATH perturb dir: " + math_perturb_dir)

# 2. Add it to Python's search path
if math_perturb_dir not in sys.path:
    sys.path.append(math_perturb_dir)

# from eval.eval_MATH import check_MATH500

data_list = []
with open(INPUT_FILE, 'r') as f:
    for line in f:
        item = json.loads(line)
        # In RL, "prompt" is the question, "ground_truth" is the answer string
        # We grab the answer from your 'chosen' field using simple logic or if you stored it separately
        # Assuming you have the correct answer stored or extractable:
        # ground_truth_unformatted = item.get('chosen') # Ensure your JSONL has this!
        # ground_truth = None
        # if ground_truth_unformatted:
        #     # Try to parse and verify the answer
        #     parsed_answer = parse(ground_truth_unformatted) # array
        #     if parsed_answer is not None:
        #         ground_truth = ', '.join(map(str, parsed_answer))
        #     else:
        #         # Fallback: use MATH500 checker
        #         is_correct, extracted_answer = check_MATH500(item['prompt'], ground_truth_unformatted)
        #         if is_correct:
        #             ground_truth = str(extracted_answer)

        ground_truth = item.get('chosen')  # ground truth is processed in the reward checker
        
        if ground_truth:
            data_list.append({
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant. Think step by step."},
                    {"role": "user", "content": item['prompt']}
                ],
                "ability": "aime", # This tag isn't strictly used by logic, just metadata
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"split": "train"}
            })

df = pd.DataFrame(data_list)
train_df = df.sample(frac=0.9, random_state=1234)
test_df = df.drop(train_df.index)

train_df.to_parquet(os.path.join(OUTPUT_PATH, "train.parquet"))
test_df.to_parquet(os.path.join(OUTPUT_PATH, "test.parquet"))