# Cell 3: AIME-Adapted Pipeline Code

# --- NEW: AIME-Specific Verification Function ---
import re

# Cell 2: Imports and Authentication
import os
import re
import json
import logging
from tqdm.notebook import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import load_dataset
import textgrad as tg
from kaggle_secrets import UserSecretsClient
import huggingface_hub

import re

def extract_aime_answer_from_solution_v2(solution_text: str) -> str | None:
    """
    Extracts the final integer answer from a detailed AIME solution string.
    This version robustly handles multiple \boxed{} and "answer is" occurrences
    by prioritizing the LAST match found for each pattern.

    Args:
        solution_text (str): The full text of the solution, including LaTeX.

    Returns:
        A string containing the integer answer (e.g., "123"), or None if no valid answer is found.
    """
    
    # Priority 1: Search for all \boxed{...} LaTeX commands.
    # The final answer is the last one.
    boxed_matches = re.findall(r'\\boxed\{(\d{1,3})\}', solution_text)
    if boxed_matches:
        # Return the last number found inside a \boxed{}
        return boxed_matches[-1]

    # Priority 2: Search for all common "answer is" type phrases.
    # We prioritize the last occurrence as it's most likely the final conclusion.
    answer_is_patterns = [
        r'(?i)(?:the|our|final) answer is:?.*?(\d{1,3})',
        r'(?i)the value is:?.*?(\d{1,3})'
    ]
    all_answer_is_matches = []
    for pattern in answer_is_patterns:
        # Find all non-overlapping matches for the pattern in the string.
        matches = re.findall(pattern, solution_text, re.DOTALL)
        if matches:
            all_answer_is_matches.extend(matches)
    
    if all_answer_is_matches:
        # Return the last number captured by any of these phrases.
        return all_answer_is_matches[-1]

    # Priority 3 (Fallback): Find the last integer in the string that is a valid AIME answer (0-999).
    # This logic remains the same as it already finds all integers and checks the last valid one.
    all_integers = re.findall(r'\d+', solution_text)
    
    for num_str in reversed(all_integers):
        num = int(num_str)
        if 0 <= num <= 999:
            return num_str
            
    return None


def verify_aime_answer(generated_solution: str, correct_answer: str) -> bool:
    """
    Verifies an AIME solution using the robust v2 extraction.
    """
    # Use the new extraction function to get the answer from the generated text
    extracted_ans_str = extract_aime_answer_from_solution_v2(generated_solution)
    
    if extracted_ans_str is None:
        return False
        
    extracted_answer = int(extracted_ans_str)
    correct_answer_int = int(correct_answer)
    
    return extracted_answer == correct_answer_int

# --- LLM Agent Classes ---
class OpenAIAgent:
    """
    A simple and reusable wrapper for making calls to the OpenAI API.
    An 'Agent' is defined by the system prompt that dictates its behavior and role.
    """
    def __init__(self, model_name: str, system_prompt: str):
        """
        Initializes the agent with a specific model and a system prompt.

        Args:
            model_name (str): The identifier for the OpenAI model to use (e.g., "o1-preview", "gpt-4o").
            system_prompt (str): The instruction that defines the agent's personality and task.
                                 This is the most important parameter for specializing the agent.
        """
        self.model = model_name
        self.system_prompt = system_prompt
        
        # Initialize the official OpenAI client.
        # It will automatically look for the "OPENAI_API_KEY" environment variable.
        self.client = OpenAI()

    def invoke(self, user_prompt: str, temperature: float = 0.5) -> str | None:
        """
        Makes a single, stateless call to the OpenAI Chat Completions API.

        Args:
            user_prompt (str): The user's direct question or instruction for this specific call.
            temperature (float): The creativity of the response. Lower is more deterministic.

        Returns:
            A string containing the model's response, or None if the API call fails.
        """
        # The 'messages' list is the standard format for the Chat Completions API.
        # It always contains the system prompt first, followed by the user prompt.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Make the actual API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            # Extract the content from the first choice in the response
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # Basic error handling for network issues, invalid keys, etc.
            logging.error(f"OpenAI API call failed for model {self.model}: {e}")
            return None


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
        logging.info(f"Proposer model loaded on {self.device}")
        # --- NEW: Define the stop sequence and get its token ID ---
        self.stop_sequence = "<|END_OF_SOLUTION|>"
        # We need the token IDs for the stopping criteria
        self.stop_token_ids = self.tokenizer.encode(self.stop_sequence, add_special_tokens=False)
        
        # The eos_token_id is a more direct way to stop if your stop sequence is a single token.
        # For multi-token sequences, StoppingCriteria is more robust.
        # For this example, we'll add it to the generation call.
    
    def solve(self, question: str) -> str:
        system_prompt = (
            """You are an expert mathematician solving a problem from the American Invitational Mathematics Examination (AIME). Provide a rigorous, step-by-step proof. Your final answer must be an integer between 0 and 999.
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
# You are an expert mathematician solving a problem from the American Invitational Mathematics Examination (AIME). Provide a rigorous, step-by-step proof. Your final answer must be an integer between 0 and 999.
#             Format your final answer as following: [[FINAL ANSWER]]
class TextGradCorrector:
    """TextGrad Corrector (O1) - with updated prompts for AIME-level critique."""
    def __init__(self, oracle_model: str):
        logging.info(f"Initializing TextGrad Corrector with oracle model: {oracle_model}")
        tg.set_backward_engine(oracle_model, override=True)
        self.optimizer_engine = tg.get_engine(oracle_model)

    def correct(self, question: str, flawed_solution: str) -> str | None:
        solution_variable = tg.Variable(
            flawed_solution,
            requires_grad=True,
            # --- MODIFIED: Role description is more specific ---
            role_description=f"A mathematical proof for the AIME problem: '{question}'"
        )
        
        # --- MODIFIED: Loss instruction is more sophisticated ---
        evaluation_instruction = (
            f"You are a math competition judge evaluating a proposed solution to the following AIME problem: '{question}'. "
            "Do not solve it yourself. Your task is to be a highly critical reviewer. "
            "Identify the primary flaw in the logical reasoning, theorem application, or algebraic manipulation. "
            "Provide a concise, specific critique of the single most important error."
        )
        loss_fn = tg.TextLoss(evaluation_instruction)
        optimizer = tg.TGD(parameters=[solution_variable], engine=self.optimizer_engine)

        try:
            loss = loss_fn(solution_variable)
            print(f"TextGrad Critique (Gradient): {loss.value}")
            loss.backward()
            optimizer.step()
            print("Optimised solution: ", solution_variable.value)
            return solution_variable.value
        except Exception as e:
            print(f"TextGrad correction step failed: {e}")
            return None

# --- NEW: Distractor Optimization Class ---
class DistractorOptimizer:
    """
    Uses TextGrad to refine a distractor within a problem to make it more challenging
    for a target LLM, without changing the problem's ground-truth answer.
    """
    def __init__(self, oracle_model: str):
        logging.info(f"Initializing TextGrad Distractor Optimizer with oracle: {oracle_model}")
        # The oracle model is used for both critiquing and rewriting
        tg.set_backward_engine(oracle_model, override=True)
        self.optimizer_engine = tg.get_engine(oracle_model)

    def optimize(self, core_question: str, original_distractor: str, correct_answer: str, successful_reasoning_trace: str) -> str:
        """
        Performs one step of distractor optimization.

        Args:
            core_question (str): The part of the prompt that is immutable.
            original_distractor (str): The text to be optimized.
            correct_answer (str): The ground-truth answer.
            successful_reasoning_trace (str): The model's successful CoT.
        
        Returns:
            A new, hopefully more challenging, distractor.
        """
        # 1. The distractor is the variable we want to change.
        distractor_variable = tg.Variable(
            original_distractor,
            requires_grad=True,
            role_description="A piece of distracting, irrelevant information embedded in a math problem."
        )

        # 2. The "Reverse" Loss Function: Critique the model's SUCCESS.
        # The goal is to generate a gradient that explains WHY the distractor failed to mislead.
        loss_instruction = (
            "The following is a math problem, a model's correct reasoning, and the specific distractor text it was given. "
            "The model successfully ignored the distractor. Your task is to critique the *distractor text*. "
            "Explain why it was not confusing enough and suggest how it could be made more salient or thematically integrated to trick the model into using it. "
            "Be very specific in your feedback on the distractor."
            f"\n\n--- CORE QUESTION ---\n{core_question}\nAnswer should be {correct_answer}."
            f"\n\n--- MODEL'S CORRECT REASONING ---\n{successful_reasoning_trace}"
        )
        loss_fn = tg.TextLoss(instruction=loss_instruction)
        
        # 3. Define the optimizer
        optimizer = tg.TGD(parameters=[distractor_variable], engine=self.optimizer_engine)

        try:
            # 4. Run the optimization cycle
            loss = loss_fn(distractor_variable) # This calls the backward engine to generate the critique
            logging.info(f"TextGrad Distractor Critique (Gradient): {loss.value}")
            loss.backward() # Propagate the critique to the variable's .grad
            optimizer.step() # Rewrite the distractor based on the critique
            
            return distractor_variable.value
        except Exception as e:
            logging.error(f"TextGrad distractor optimization failed: {e}")
            return original_distractor # Return original on failure

# --- The Main Pipeline Class (logic is the same, but uses new agents/verification) ---
max_distractor_optim_steps = 3
class AdversarialPipeline:
    def __init__(self, proposer_model_name: str, oracle_model_name: str):
        self.proposer = HuggingFaceAgent(proposer_model_name)
        self.corrector = TextGradCorrector(oracle_model=oracle_model_name)
        self.distractor_optimizer = DistractorOptimizer(oracle_model=oracle_model_name)
        self.referee = OpenAIAgent(oracle_model_name, "You are an objective expert. Provide a correct, step-by-step solution. Your final answer must be correct and clearly stated.")
        self.validator = OpenAIAgent(oracle_model_name, "You are a validation expert. Your task is to check if a math problem is logical and well-formed. Respond with 'VALID' or 'INVALID' and a brief explanation.")
        self.max_optim_steps = max_distractor_optim_steps
        
        

    def generate_single_example(self, original_problem: dict):
        question = original_problem['problem'] # Dataset key is 'problem'
        # --- MODIFIED: AIME solution is a full trace, not just a number ---
        ground_truth_solution = original_problem['solution'] 

        
        
        # We need the final integer answer for verification.
        # We'll get it from the ground truth solution trace using our verifier.
        correct_answer_str = str(int(extract_aime_answer_from_solution_v2(ground_truth_solution)))
        print(question, ground_truth_solution, correct_answer_str)
        initial_solution = self.proposer.solve(question)
        if not initial_solution: return None
        print(initial_solution)
        # --- MODIFIED: Use the new verification function ---
        if verify_aime_answer(initial_solution, correct_answer_str):
            print("SKIPPING")
            print("Proposer solved AIME problem correctly. Skipping.")
            return None

        print(f"Proposer failed. Initial Solution:\n---\n{initial_solution}\n---")
        
        y_chosen = self.corrector.correct(question, initial_solution)
        y_rejected = initial_solution
        print(y_chosen, y_rejected)
        if not y_chosen: return None
        
        # Validate that the new corrected solution is correct.
        if verify_aime_answer(y_chosen, correct_answer_str):
            print(f"Successfully generated corrected preference pair for AIME problem.")
            return {"prompt": question, "chosen": y_chosen, "rejected": y_rejected}
        else:
            logging.warning("Validation failed: TextGrad's AIME solution was still incorrect.")
            return None