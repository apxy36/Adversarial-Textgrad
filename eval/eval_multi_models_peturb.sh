cd eval
echo "Evaluating Qwen-3.0-14B models fine-tuned on Metamath subsets..."
python eval_math_peturb.py --model_path "/mnt/ssd/iclrtemp/adapters/qwen3_14b_Metamath60_SFT/checkpoint-100" --output_dir "evaluation_results_qwen3_14b_Metamath100_SFT_full_peturb.json" --data_dir "MATH_Perturb/math_perturb/" --limit 100
echo "Completed evaluation for Qwen-3.0-14B Metamath60_SFT model."
python eval_math_peturb.py --model_path "/mnt/ssd/iclrtemp/adapters/qwen3_14b_Metamath120_SFT/checkpoint-100" --output_dir "evaluation_results_qwen3_14b_Metamath120_SFT_full_peturb.json" --data_dir "MATH_Perturb/math_perturb/" --limit 100
echo "Completed evaluation for Qwen-3.0-14B Metamath120_SFT model."
python eval_math_peturb.py --model_path "/mnt/ssd/iclrtemp/adapters/qwen3_14b_Metamath180_SFT/checkpoint-100" --output_dir "evaluation_results_qwen3_14b_Metamath180_SFT_full_peturb.json" --data_dir "MATH_Perturb/math_perturb/" --limit 100
echo "Completed evaluation for Qwen-3.0-14B Metamath180_SFT model."
python eval_math_peturb.py --model_path "/mnt/ssd/iclrtemp/adapters/qwen3_14b_Metamath240_SFT/checkpoint-50" --output_dir "evaluation_results_qwen3_14b_Metamath240_SFT_full_peturb.json" --data_dir "MATH_Perturb/math_perturb/" --limit 100
echo "Completed evaluation for Qwen-3.0-14B Metamath240_SFT model."
python eval_math_peturb.py --model_path "/mnt/ssd/iclrtemp/adapters/qwen3_14b_Metamath300_SFT/checkpoint-50" --output_dir "evaluation_results_qwen3_14b_Metamath300_SFT_full_peturb.json" --data_dir "MATH_Perturb/math_perturb/" --limit 100
echo "Completed evaluation for Qwen-3.0-14B Metamath300_SFT model."