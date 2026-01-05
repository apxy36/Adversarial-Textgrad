import os
import sys
# Fix memory fragmentation (Optional but recommended)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 2. Disable vLLM's experimental V1 engine (Your traceback shows you are using V1).
# We want the stable V0 engine.
os.environ["VLLM_USE_V1"] = "0"

# 3. specific fix for "custom_all_reduce" crash
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "true"

# 4. Set Visible Devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# --- DEVICE CONFIGURATION ---
# To use multiple GPUs, list them with commas (e.g., "0,1,2,3").
# If you set this to "0,1", vLLM will automatically use both GPUs via Tensor Parallelism.
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # <--- MODIFY THIS LIST AS NEEDED

# vLLM Optimizations
# If you run into OutOfMemory errors on initialization, lower the utilization (default is 0.90)
os.environ["gpu_memory_utilization"] = "0.95" 


import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class VLLMAgent:
    def __init__(self, model_path: str, prompt: str = ""):
        # Auto-detect number of GPUs
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        tp_size = len(visible_devices.split(","))
        print(f"Initializing vLLM on {tp_size} GPU(s)...")

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 2. Load vLLM Engine
        # We DISABLE LoRA here because we are using a merged model.
        # This prevents the 'vllm.lora.models' import error.
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            dtype="float16",      # Merged model is 16-bit
            trust_remote_code=True,
            enable_lora=False,    # CRITICAL: Ensures we don't trigger LoRA logic
            gpu_memory_utilization=0.90, 
        )

        self.stop_sequence = "<|END_OF_SOLUTION|>"
        
        # Default System Prompt
        if prompt == "":
            self.prompt = (
                """Below is an instruction that describes a task. Write a rigorous and appropriate step-by-step solution to the task.
                Be concise and efficient in your reasoning.
                You MUST format and put your final answer strictly within \\boxed{}."""
            )
        else:
            self.prompt = prompt

    def solve(self, question: str) -> str:
        # Create prompt using chat template
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": question}
        ]
        
        # Convert to string (vLLM handles tokenization internally for speed)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=32768,
            stop=[self.stop_sequence, self.tokenizer.eos_token]
        )

        # Generate
        outputs = self.llm.generate([prompt_text], sampling_params)
        
        # Extract text
        generated_text = outputs[0].outputs[0].text.strip()
        
        print(generated_text)
        print("END_RESPONSE")
        return generated_text