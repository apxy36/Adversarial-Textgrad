import argparse
import json
import random
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Sample examples from MATH dataset excluding MATH500.")
    
    # Arguments for the number of samples
    parser.add_argument("--set1_size", type=int, default=100, help="Number of examples for the first JSON file.")
    parser.add_argument("--set2_size", type=int, default=100, help="Number of examples for the second JSON file.")
    
    # Arguments for output filenames
    parser.add_argument("--output1", type=str, default="math_sample_1.json", help="Filename for the first output.")
    parser.add_argument("--output2", type=str, default="math_sample_2.json", help="Filename for the second output.")

    parser.add_argument("--data_dir", type=str, default="qwedsacf/competition_math", help="Directory containing MATH dataset files.")
    
    # Argument to select which split of MATH to sample from (default is test)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], 
                        help="The split of the original MATH dataset to sample from (usually 'test').")

    args = parser.parse_args()

    print("--- Loading Datasets ---")
    # 1. Load the MATH500 dataset (The exclusion list)
    # MATH500 is typically derived from the test set, but we load it to get the forbidden strings.
    print("Loading MATH500 (HuggingFaceH4/math-500)...")
    try:
        ds_math500 = load_dataset("HuggingFaceH4/math-500", split="test")
    except Exception as e:
        print(f"Error loading MATH500: {e}")
        return

    # Create a set of problems from MATH500 for O(1) lookup
    # We strip whitespace to ensure robust matching
    forbidden_problems = set(x['problem'].strip() for x in ds_math500)
    print(f"Found {len(forbidden_problems)} unique problems in MATH500 to exclude.")

    # 2. Load the standard MATH dataset
    print(f"Loading MATH (qqq/competition_math) split: {args.split}...")
    try:
        ds_math = load_dataset(args.data_dir, split=args.split)
    except Exception as e:
        print(f"Error loading MATH: {e}")
        return

    # 3. Filter the MATH dataset
    print("Filtering MATH dataset...")
    available_pool = []
    
    for item in ds_math:
        # Check if the problem text exists in the forbidden set
        if item['problem'].strip() not in forbidden_problems: # problem
            available_pool.append(item)

    total_available = len(available_pool)
    print(f"Total examples in MATH '{args.split}' after removing MATH500: {total_available}")

    # 4. Check if we have enough data
    total_requested = args.set1_size + args.set2_size
    if total_requested > total_available:
        print(f"Error: You requested {total_requested} examples, but only {total_available} are available after filtering.")
        return

    # 5. Randomly Sample
    print(f"Sampling {args.set1_size} for Set 1 and {args.set2_size} for Set 2...")
    
    # Shuffle the pool completely
    random.shuffle(available_pool)
    
    # Slice the list to get non-overlapping sets
    dataset_1 = available_pool[:args.set1_size]
    dataset_2 = available_pool[args.set1_size : args.set1_size + args.set2_size]

    # 6. Save to JSON
    print(f"Saving to {args.output1}...")
    with open(args.output1, 'w', encoding='utf-8') as f:
        json.dump(dataset_1, f, indent=4, ensure_ascii=False)

    print(f"Saving to {args.output2}...")
    with open(args.output2, 'w', encoding='utf-8') as f:
        json.dump(dataset_2, f, indent=4, ensure_ascii=False)

    print("--- Done ---")

if __name__ == "__main__":
    main()