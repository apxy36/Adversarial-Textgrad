

import  os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.conv.fp32_precision = 'tf32'

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# from transformers import BitsAndBytesConfig
# quant_config = BitsAndBytesConfig(load_in_16bit=True,)

from transformers import LogitsProcessor

class ClampLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # scores = logits (batch, vocab)
        scores = scores.clone()
        # replace non-finite with a very negative number
        scores[~torch.isfinite(scores)] = -1e9
        # optionally clamp extreme values to avoid overflow in softmax
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        return scores

# pass it to generate
from transformers import LogitsProcessorList
proc_list = LogitsProcessorList([ClampLogitsProcessor()])


class HuggingFaceAgent:
    """Proposer model (Qwen3-4B) - with updated prompt for AIME."""
    # In your HuggingFaceAgent class

    def __init__(self, model_name: str, prompt: str = ""):
        # ... (previous init code) ...
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="cuda:1",
            trust_remote_code=True
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="cuda:1",
        pad_token_id=self.tokenizer.eos_token_id,  # Force padding to equal EOS
        # ADD THIS LINE:
        # model_kwargs={"attn_implementation": "eager", "quantization_config": quant_config}
        )
        print(f"Proposer model loaded on {self.device}")
        # --- NEW: Define the stop sequence and get its token ID ---
        self.stop_sequence = "<|END_OF_SOLUTION|>"
        # We need the token IDs for the stopping criteria
        self.stop_token_ids = self.tokenizer.encode(self.stop_sequence, add_special_tokens=False)
        if prompt == "":
            self.prompt = (
                """You are an expert mathematician solving a problem from a math competition. Provide a rigorous, step-by-step solution to the problem.
                Be concise and efficient in your reasoning.
                You MUST format and put your final answer strictly within \\boxed{}."""
            )
        else:
            self.prompt = prompt
        
        # The eos_token_id is a more direct way to stop if your stop sequence is a single token.
        # For multi-token sequences, StoppingCriteria is more robust.
        # For this example, we'll add it to the generation call.
    
    def solve(self, question: str) -> str:
        
        
        
        messages = [{"role": "system", "content": self.prompt}, {"role": "user", "content": question}]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Check for the "Silent Killer": Vocab Mismatch
        # If the tokenizer produces an ID larger than the model knows, it reads garbage (NaNs).
        # if hasattr(self.pipe, 'tokenizer') and hasattr(self.pipe, 'model'):
        #     t_vocab = len(self.pipe.tokenizer)
        #     m_vocab = self.pipe.model.config.vocab_size
        #     print(f"DEBUG: Tokenizer Size: {t_vocab} | Model Vocab Size: {m_vocab}")
        #     if t_vocab > m_vocab:
        #         print("CRITICAL ERROR DETECTED: Tokenizer is larger than Model. This causes NaNs.")
        # --- DEBUG BLOCK END ---
        
        # --- MODIFIED: Add stopping criteria to the pipeline call ---
        outputs = self.pipe(
            prompt,
            max_new_tokens=1536*40, # Keep a generous max limit as a fallback
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            # truncation=True,
            # This is the key parameter. Generation stops when this token is produced.
            # eos_token_id=self.stop_token_ids + [self.pipe.tokenizer.eos_token_id],
            pad_token_id=self.pipe.tokenizer.eos_token_id,
            # logits_processor=proc_list,
        )
        
        generated_text = outputs[0]['generated_text'][len(prompt):].strip()
        print(outputs[0])
        # --- NEW: Clean up the output by removing the stop sequence ---
        if generated_text.endswith(self.stop_sequence):
            generated_text = generated_text[:-len(self.stop_sequence)].strip()
        print("END_RESPONSE")
        return generated_text