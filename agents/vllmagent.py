import os
import sys
# Fix memory fragmentation (Optional but recommended)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

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
GPU_MEM_UTIL = 0.7  # Adjust based on your GPU memory capacity
# If you run into OutOfMemory errors on initialization, lower the utilization (default is 0.90)
os.environ["gpu_memory_utilization"] = str(GPU_MEM_UTIL) 


# import torch
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams

import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# If you are using runtime LoRA (Option 2), uncomment the import below:
# from vllm.lora.request import LoRARequest

class VLLMAgent:
    def __init__(self, model_path: str, prompt: str = "", think: bool = True, use_lora: bool = True):
        # Auto-detect number of GPUs
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        tp_size = len(visible_devices.split(","))
        print(f"Initializing vLLM on {tp_size} GPU(s)...")

        self.think = think
        
        # 1. Load Tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if not self.think:
            UPSTREAM_BASE_ID = "Qwen/Qwen3-14B-FP8"  # <--- REPLACE THIS WITH YOUR ACTUAL BASE MODEL
        else:
            UPSTREAM_BASE_ID = "openai/gpt-oss-20b"  # <--- REPLACE THIS WITH YOUR ACTUAL BASE MODEL

        if not use_lora:
            UPSTREAM_BASE_ID = model_path  # Use the provided model path if not using LoRA
        try:
            print(f"Attempting to load base tokenizer from: {model_path}")
            # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
            self.tokenizer = AutoTokenizer.from_pretrained(UPSTREAM_BASE_ID, trust_remote_code=True)
        except AttributeError:
            print(f"\n Local tokenizer config error detected. Falling back to upstream: {UPSTREAM_BASE_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(UPSTREAM_BASE_ID, trust_remote_code=True)
        except OSError:
             print(f"\n Local tokenizer not found. Falling back to upstream: {UPSTREAM_BASE_ID}")
             self.tokenizer = AutoTokenizer.from_pretrained(UPSTREAM_BASE_ID, trust_remote_code=True)


        # 2. Load vLLM Engine
        # NOTE: Make sure 'model_path' points to your MERGED BF16 model (Option 1)
        # or your Base Model if using runtime LoRA (Option 2).
        if use_lora:
            self.lora_request = LoRARequest("math_adapter", 1, model_path) 
        else:
            self.lora_request = None

        self.llm = None
        if self.lora_request:
            print(f"Loading vLLM with runtime LoRA adapter from: {model_path}")
            self.llm = LLM(
                model=UPSTREAM_BASE_ID,
                tensor_parallel_size=tp_size,
                # dtype="float16",         # Uncomment if your model is in FP16   
                trust_remote_code=True,
                gpu_memory_utilization=GPU_MEM_UTIL, # Increased slightly for batching efficiency
                tokenizer = UPSTREAM_BASE_ID,
                max_model_len = 4096,
                enable_lora=True,          # Uncomment if using runtime LoRA
                max_lora_rank=16,          # Uncomment if using runtime LoRA
                # enforce_eager=True,
            )
        else:
            self.llm = LLM(
                model=UPSTREAM_BASE_ID if self.lora_request else model_path,
                tensor_parallel_size=tp_size,
                # dtype="float16",         # Uncomment if your model is in FP16   
                trust_remote_code=True,
                gpu_memory_utilization=GPU_MEM_UTIL, # Increased slightly for batching efficiency
                tokenizer = UPSTREAM_BASE_ID,
                max_model_len = 4096,
                # enable_lora=True,          # Uncomment if using runtime LoRA
                # max_lora_rank=16,          # Uncomment if using runtime LoRA
                # enforce_eager=True,
            )
        
        # If using runtime LoRA, define request here:
        # self.lora_request = LoRARequest("math_adapter", 1, "/path/to/adapter")

        self.stop_sequence = "<|END_OF_SOLUTION|>"
        
        # Default System Prompt
        if prompt == "":
            self.prompt = (
                """Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                Solve the following problem.
                - Provide a direct final step-by-step solution in a single pass.
                - Put your final answer on its own line as \\boxed{{...}}.
                - Use exactly one \\boxed{{}} and do not box intermediate results.
                
                Output format:
                -Steps: (multiple lines)
                -Final answer: \\boxed{{...}}  (last line only)"""
            )
        else:
            self.prompt = prompt

        # if self.think:
        #     self.prompt += "\n Reasoning: low\n"

    def batch_solve(self, questions: List[str]) -> List[str]:
        """
        Runs inference on a batch of questions.
        """
        # 1. Prepare Structured Inputs
        # vLLM expects a list of dictionaries for pre-tokenized inputs:
        # inputs = [{"prompt_token_ids": [1, 2, ...]}, ...]
        inputs: List[Dict[str, Any]] = []
        
        for question in questions:
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": question}
            ]
            # if self.think:
                # messages[0]['content'] ="\n Reasoning: low\n" + messages[0]['content']
            
            # Prepare kwargs
            kwargs = {"tokenize": True, "add_generation_prompt": True}
            if not self.think:
                kwargs["enable_thinking"] = False
            else:
                kwargs["reasoning_effort"] = 'low'
            # Get token IDs (List[int])
            token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
            
            # Wrap in dictionary (The Fix)
            inputs.append({"prompt_token_ids": token_ids})

        # 2. Sampling parameters
        if self.think:
            sampling_params = SamplingParams(
                temperature=1.0, # 0.7
                top_p=1.0, # 0.8
                max_tokens = 4096,
                top_k=0, # 20
                # min_p=0.0, 
                stop=[self.stop_sequence, self.tokenizer.eos_token]
            )
        else:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                max_tokens = 4096,
                top_k=20,
                min_p=0.0,
                stop=[self.stop_sequence, self.tokenizer.eos_token]
            )

        # 3. Generate
        # We pass the list of dicts to the first argument ('prompts')
        if self.lora_request:
            outputs = self.llm.generate(
                prompts=inputs,
                sampling_params=sampling_params,
                use_tqdm=True,
                lora_request=self.lora_request  # Uncomment if using runtime LoRA
            )
        else:
            outputs = self.llm.generate(
                prompts=inputs,
                sampling_params=sampling_params,
                use_tqdm=True
            )
        
        # 4. Extract Results
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append(generated_text)
            
        return results


    # Keep single solve for compatibility, wrapper around batch
    def solve(self, question: str) -> str:
        return self.batch_solve([question])[0]