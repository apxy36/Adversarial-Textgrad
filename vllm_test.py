import os
import sys

# --- 1. CONFIGURATION ---
# We force these settings to ensure stability on your hardware.
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 

from vllm import LLM, SamplingParams

def main():
    print(">>> Initializing vLLM...")

    # --- 2. INITIALIZE ---
    # We pass 'disable_custom_all_reduce' to stop the communication crash.
    # We pass 'enforce_eager' to stop the "Launch Failure" crash.
    llm = LLM(
        model="Qwen/Qwen3-14B",
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        disable_custom_all_reduce=True, # <--- Fixes "invalid argument"
        enforce_eager=True,             # <--- Fixes "launch failure" / V1 stability
    )

    print(">>> Model Loaded!")
    
    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=20)
    
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Result: {output.outputs[0].text}")

if __name__ == "__main__":
    import multiprocessing
    # 'spawn' is mandatory for vLLM with CUDA
    multiprocessing.set_start_method('spawn', force=True)
    main()