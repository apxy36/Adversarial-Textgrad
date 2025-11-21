
# Cell 2: Imports and Authentication
import os
import re
import json
import logging
from tqdm.notebook import tqdm

from agents.openaiagent import OpenAIAgent
from agents.huggingface import HuggingFaceAgent
from agents.problem_optimiser import AdversarialProblemOptimizer
# from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import load_dataset
import textgrad as tg
import textgrad.engine as tge
from textgrad.engine.openai import ChatOpenAI
# from kaggle_secrets import UserSecretsClient
import huggingface_hub



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

def verify_gsm8k_answer(generated_solution: str, correct_answer: str) -> bool:
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

def extract_answer_from_gsm8k(text):
    try:
        start_idx = text.find("#### ")
        ans = text[start_idx + 5:].replace(" ", "")
    except:
        ans = "-1"
    return ans


# You are an expert mathematician solving a problem from the American Invitational Mathematics Examination (AIME). Provide a rigorous, step-by-step proof. Your final answer must be an integer between 0 and 999.
#             Format your final answer as following: [[FINAL ANSWER]]
class TextGradCorrector:
    """TextGrad Corrector (O1) - with updated prompts for AIME-level critique."""
    def __init__(self, oracle_model: str):
        print(f"Initializing TextGrad Corrector with oracle model: {oracle_model}")
        oracle_model_LLM = ChatOpenAI(model_string=oracle_model, is_multimodal=tge._check_if_multimodal(oracle_model))
        tg.set_backward_engine(oracle_model_LLM, override=True)
        # ChatOpenAI(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
        
        self.optimizer_engine = ChatOpenAI(model_string=oracle_model, is_multimodal=tge._check_if_multimodal(oracle_model))
        # tg.get_engine(oracle_model)

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
        print(f"Initializing TextGrad Distractor Optimizer with oracle: {oracle_model}")
        # The oracle model is used for both critiquing and rewriting
        oracle_model_LLM = ChatOpenAI(model_string=oracle_model, is_multimodal=tge._check_if_multimodal(oracle_model))
        tg.set_backward_engine(oracle_model_LLM, override=True)
        self.optimizer_engine = ChatOpenAI(model_string=oracle_model, is_multimodal=tge._check_if_multimodal(oracle_model))
        # tg.get_engine(oracle_model)

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
            print(f"TextGrad Distractor Critique (Gradient): {loss.value}")
            loss.backward() # Propagate the critique to the variable's .grad
            optimizer.step() # Rewrite the distractor based on the critique
            
            return distractor_variable.value
        except Exception as e:
            print(f"TextGrad distractor optimization failed: {e}")
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
        # initial_solution = self.proposer.solve(question)
        initial_solution = "placeholder solution"
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
            print("Validation failed: TextGrad's AIME solution was still incorrect.")
            return None
        

def new_generate_from_single_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature=1,
        max_tokens=2000,
        top_p=0.99,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            # max_tokens=max_tokens,
            # top_p=top_p,
        )

        response = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

ChatOpenAI._generate_from_single_prompt = new_generate_from_single_prompt

# --- The Main Iterative Pipeline Class ---
# Assume OpenAIAgent and other helpers are defined elsewhere



class IterativeHardeningPipeline:
    def __init__(self, proposer_model_name: str, oracle_model_name: str, max_iterations: int = 5):
        self.proposer = HuggingFaceAgent(proposer_model_name)
        self.optimizer = AdversarialProblemOptimizer(oracle_model=oracle_model_name)
        self.validator = OpenAIAgent(oracle_model_name, "You are a validation expert. Check if a math problem is logically coherent and has one unambiguous answer. Respond with 'VALID' or 'INVALID' and a brief explanation.")
        self.max_iterations = max_iterations
        self.distractor_proposer = OpenAIAgent(oracle_model_name, "You are a math expert. Add a new, confusingly-worded but logically superfluous statement (distractor) to a math problem to increase cognitive load, without changing the final answer.")

        # test openai
        test = self.distractor_proposer.invoke("What is 2 + 2?")
        print("Openai Validator test: ", test)

    def process_problem(self, seed_problem: dict):
        # Adapt seed problem structure for our optimizer (example for GSM8k)
        ground_truth_solution = extract_answer_from_gsm8k(seed_problem['answer']) # solution
        problem = {
            'framing': "", # GSM8k has no separate framing
            'core_facts': seed_problem['question'], # problem
            'question': "", # The question is part of the core_facts
            'answer': int(ground_truth_solution), # int(extract_aime_answer_from_solution_v2(ground_truth_solution)),
            'full_problem_text': seed_problem['question']
        }
        
        if not problem['answer']:
            print("Could not extract ground truth answer from seed problem. Skipping.")
            return None

        current_prompt = problem['full_problem_text']
        reasoning_traces = [] # To collect the traces at each difficulty level
        distractor = ""
        answers = []
        problems= [current_prompt]
        
        for i in range(self.max_iterations):
            
            
            print(f"--- Iteration {i+1}/{self.max_iterations} for problem ---")
            
            solution_trace = self.proposer.solve(current_prompt)
            # solution_trace = "placeholder trace for testing"
            if not solution_trace:
                print("Proposer failed to generate a valid solution trace.")
                # We can't proceed if the model gives no output.
                return None
            
            reasoning_traces.append({"prompt": current_prompt, "trace": solution_trace})
            extracted_ans_str = "-1"
            try:
                extracted_ans_str = extract_aime_answer_from_solution_v2(solution_trace)
            except:
                extracted_ans_str = "-1"
                print("error, cannot extract ans from trace")
            if extracted_ans_str == None:
                extracted_ans_str = "-1"
            answers.append(int(extracted_ans_str))
            if not verify_gsm8k_answer(solution_trace, str(problem['answer'])):
                print("SUCCESS: Proposer failed! Adversarial example found.")
                # This is our final, high-quality data point.
                
                return {
                    "seed_problem": seed_problem['question'], # problem
                    "final_hardened_problem": current_prompt,
                    "correct_answer": problem['answer'],
                    "proposer_reasoning_traces": reasoning_traces, # Contains all attempts
                    "proposer_answers": answers,
                    "final_failed_trace": solution_trace,
                        "problems": problems,
                }
            
            print("Proposer succeeded. Attempting to harden the problem...")

            max_tries = 3
            num_tries = 0
            soln_valid = False
            while not soln_valid:
                print("distractor try ", num_tries)
                if i == 0:
                    distractor = self.distractor_proposer.invoke(f"""
                    Given the following problem, successful solution trace and answer, generate distractor text. 
                    Problem: {current_prompt}
                    Successful solution trace: {solution_trace}
                    Answer: {problem['answer']}
                                                                 """)
                    print("Init distractor: ", distractor)
                # Proposer succeeded, so we harden the problem for the next iteration
                output = self.optimizer.harden_problem(problem, solution_trace, distractor, problem['answer'])
                hardened_prompt = output[0]
                distractor = output[1]
                print("New distractor: ", distractor)
                # --- Validation is critical ---
                if not hardened_prompt:
                    print("Hardening process failed to produce a new problem. Stopping iteration.")
                    return None
                    
                validation_response = self.validator.invoke(
                    f"Is the following provided problem valid and unambiguous, and is its answer the same as the one provided? \n\n Provided problem: {hardened_prompt} \n Provided Answer: {ground_truth_solution} For reference, the original problem, which the provided problem was inspired by, was: {problems[0]}. The provided problem has been modified to be more difficult, but should still have the same answer as the original problem.",
                    temperature=1.0
                )
                print("Validation: ", validation_response)
                if "INVALID" in validation_response.upper():
                    print(f"Generated problem failed validation. Stopping iteration. Reason: {validation_response}")
                    num_tries += 1
                else:
                    soln_valid = True

                if num_tries >= max_tries:
                    return None
            # add retry pipeline
            
            # Update the prompt for the next loop
            current_prompt = hardened_prompt
            problem['full_problem_text'] = hardened_prompt # Update context for next optimization
            print("Harder problem: ", hardened_prompt)
            problems.append(hardened_prompt)
        print("Max iterations reached, but proposer kept succeeding. No final failure example generated.")
        return None