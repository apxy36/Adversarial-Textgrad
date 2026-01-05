import os
import sys

# --- DEVICE CONFIGURATION ---
# Unsloth handles devices best via CUDA_VISIBLE_DEVICES. 
# We set this to "1" to match your original "cuda:1" requirement.


# Memory fragmentation settings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from transformers import StoppingCriteria, StoppingCriteriaList

class StringStopper(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        self.stop_string = stop_string
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last few tokens to check for the string
        # We only check the newly generated part to be fast
        decoded_text = self.tokenizer.decode(input_ids[0][-10:]) 
        return self.stop_string in decoded_text

# Usage


# --- OPTIMIZATION SETTINGS ---
# Unsloth requires Flash Attention. Do NOT disable it.
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.conv.fp32_precision = 'tf32'

class ClampLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # scores = logits (batch, vocab)
        scores = scores.clone()
        # replace non-finite with a very negative number
        scores[~torch.isfinite(scores)] = -1e9
        # optionally clamp extreme values to avoid overflow in softmax
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        return scores

class UnslothAgent:
    """Proposer model using Unsloth FastLanguageModel."""

    def __init__(self, model_name: str, prompt: str = "", think: bool = True):
        self.device = "cuda" # Unsloth automatically uses the visible CUDA device
        self.think = think
        # Unsloth requires a max_seq_length. 
        # Your original code requested 1536*40 (~61k tokens), so we set a high limit.
        self.max_seq_length = 32768 
        
        print(f"Loading Unsloth model: {model_name} on {self.device}...")
        
        # Load Model and Tokenizer via Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = self.max_seq_length,
            dtype = None,        # None = Auto detection (usually bfloat16 for Ampere+)
            load_in_4bit = False, # Change to False if you want 16bit. True is significantly faster/lighter.
            trust_remote_code = True,
        )

        self.stopper = StringStopper("<|im_end|>", self.tokenizer)

        # ENABLE INFERENCE MODE (Crucial for Unsloth speedup)
        FastLanguageModel.for_inference(self.model)
        
        print(f"Proposer model loaded.")

        # --- Define Stop Sequence ---
        self.stop_sequence = "<|END_OF_SOLUTION|>"
        # We need the token IDs for the stopping criteria
        # Note: Unsloth uses the underlying HF tokenizer, so this works the same.
        self.stop_token_ids = self.tokenizer.encode(self.stop_sequence, add_special_tokens=False)

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

    def solve(self, question: str) -> str:
        
        messages = [{"role": "system", "content": self.prompt}, {"role": "user", "content": question}]
        
        # Apply template
        # Unsloth's tokenizer is compatible with standard HF chat templates
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        if not self.think:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True,
                return_tensors = "pt",
                enable_thinking = False
            ).to("cuda")

        # --- GENERATION ---
        # We use model.generate directly instead of pipeline for better Unsloth integration
        
        # Optional: Add the Clamp Processor if specifically needed (commented out as per original)
        # logits_processor = LogitsProcessorList([ClampLogitsProcessor()])
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids = inputs,
                max_new_tokens = 4096, # Large buffer as requested (1536*40 approx)
                do_sample = True,
                temperature = 0.7,
                top_p = 0.8,
                top_k = 20,
                min_p = 0.0,
                use_cache = True,
                pad_token_id = self.tokenizer.eos_token_id,
                # stopping_criteria=StoppingCriteriaList([self.stopper]), # <--- FORCE STOP MMIQC
                # eos_token_id = self.stop_token_ids, # Uncomment if you want hard stopping on the token
                # logits_processor = logits_processor
            )

        # Decode output
        # Unsloth/HF generate returns [Prompt + Response], so we slice off the prompt
        generated_ids = outputs[0][inputs.shape[1]:] 
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # --- Post-processing ---
        if generated_text.endswith(self.stop_sequence):
            generated_text = generated_text[:-len(self.stop_sequence)].strip()
            
        # Debug printing similar to original
        print(generated_text) 
        print("END_RESPONSE")
        
        return generated_text