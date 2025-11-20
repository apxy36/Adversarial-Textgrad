from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class HuggingFaceAgent:
    """Proposer model (Qwen3-4B) - with updated prompt for AIME."""
    # In your HuggingFaceAgent class

    def __init__(self, model_name: str):
        # ... (previous init code) ...
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
        print(f"Proposer model loaded on {self.device}")
        # --- NEW: Define the stop sequence and get its token ID ---
        self.stop_sequence = "<|END_OF_SOLUTION|>"
        # We need the token IDs for the stopping criteria
        self.stop_token_ids = self.tokenizer.encode(self.stop_sequence, add_special_tokens=False)
        
        # The eos_token_id is a more direct way to stop if your stop sequence is a single token.
        # For multi-token sequences, StoppingCriteria is more robust.
        # For this example, we'll add it to the generation call.
    
    def solve(self, question: str) -> str:
        system_prompt = (
            """You are an expert mathematician solving a problem from a math competition. Provide a rigorous, step-by-step proof. 
            Be concise and efficient in your reasoning. 
            You MUST format and include your final answer strictly as following: [[FINAL ANSWER]]
            End your entire response with the exact phrase: <|END_OF_SOLUTION|>"""
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # --- MODIFIED: Add stopping criteria to the pipeline call ---
        outputs = self.pipe(
            prompt,
            max_new_tokens=1536*40, # Keep a generous max limit as a fallback
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            # This is the key parameter. Generation stops when this token is produced.
            # eos_token_id=self.stop_token_ids + [self.pipe.tokenizer.eos_token_id],
            pad_token_id=self.pipe.tokenizer.eos_token_id
        )
        
        generated_text = outputs[0]['generated_text'][len(prompt):].strip()
        print(outputs[0])
        # --- NEW: Clean up the output by removing the stop sequence ---
        if generated_text.endswith(self.stop_sequence):
            generated_text = generated_text[:-len(self.stop_sequence)].strip()
        print("END_RESPONSE")
        return generated_text