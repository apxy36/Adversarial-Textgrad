
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
import textgrad.engine as tge
from textgrad.engine.openai import ChatOpenAI
# from kaggle_secrets import UserSecretsClient
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

class AdversarialProblemOptimizer:
    """
    A generalized framework to increase the difficulty of a given problem
    by applying a dynamically selected, TextGrad-powered adversarial strategy.
    """
    def __init__(self, oracle_model: str):
        """
        Initializes the optimizer with a powerful oracle model and registers
        the available adversarial strategies.
        """
        self.oracle_model_llm = ChatOpenAI(model_string=oracle_model, is_multimodal=tge._check_if_multimodal(oracle_model))
        tg.set_backward_engine(self.oracle_model_llm, override=True)
        
        # Initialize the meta-agent for strategy selection
        self.selector = OpenAIAgent(
            oracle_model,
            "You are a red team strategist. Your job is to analyze how a model solved a problem and choose the best strategy from a list to create a new, harder problem that will make it fail."
        )
        
        # Initialize the TextGrad engines
        # tg.set_backward_engine(oracle_model, override=True)
        self.optimizer_engine = ChatOpenAI(model_string=oracle_model, is_multimodal=tge._check_if_multimodal(oracle_model))
        
        # --- The Core of the General Framework: The Strategy Registry ---
        self.strategies = {}
        self._register_strategies()

    def _register_strategies(self):
        """
        Defines and registers the available adversarial strategies.
        Each strategy is a dictionary defining its TextGrad recipe.
        This is where you would add new failure modes.
        """
        
        # STRATEGY 1: ADVERSARIAL REFRAMING
        self.strategies["Adversarial Reframing"] = {
            "description": "Rewrite the problem's narrative to resemble a famous, complex paradox to trick the model into overthinking.",
            "variable_extractor": lambda p: p.get('framing'),
            "loss_instruction_template": (
                "A model correctly solved a problem by ignoring its simple framing. "
                "Critique this framing. Explain why it was too direct. Suggest how to rewrite it to mimic a more complex problem type (like a probability puzzle) "
                "to trick a pattern-matching model into applying an incorrect, complex algorithm. Rationale: '{rationale}'"
                "\n\n--- UNCHANGING PART OF PROBLEM ---\n{context}"
            ),
            "reconstructor": lambda p, new_var: f"{new_var} {p.get('core_facts', '')} {p['question']}"
        }
        
        # STRATEGY 2: SPURIOUS FEATURE AMPLIFICATION
        self.strategies["Spurious Feature Amplification"] = {
            "description": "Amplify the language of an irrelevant feature in structured data to make it sound more important and causally linked to the answer.",
            "variable_extractor": lambda p: p.get('spurious_feature_text'),
            "loss_instruction_template": (
                "A model correctly ignored a spurious feature in a structured problem. "
                "Critique this feature's text. Rewrite it to use more scientific or causal-sounding language to make it a more tempting feature to use. Rationale: '{rationale}'"
                "\n\n--- FULL PROBLEM CONTEXT ---\n{context}"
            ),
            "reconstructor": lambda p, new_var: p['full_problem_text'].replace(p['spurious_feature_text'], new_var)
        }
        
        # STRATEGY 3: REDUNDANT CONSTRAINT INJECTION
        self.strategies["Redundant Constraint Injection"] = {
            "description": "Add a new, confusingly-worded but logically superfluous clue to a logic puzzle to increase cognitive load.",
            "variable_extractor": lambda p: "[PLACEHOLDER_FOR_A_NEW_CONFUSING_CLUE]",
            "loss_instruction_template": (
                "A model solved this logic puzzle efficiently. Your task is to generate a new clue. "
                "This clue MUST NOT add new information but should be logically redundant with the existing clues, forcing the model to re-verify its work. "
                "Make it complexly worded. Rationale: '{rationale}'"
                "\n\n--- EXISTING CLUES ---\n{context}"
            ),
            "reconstructor": lambda p, new_var: f"{p['full_problem_text']}\n- New Clue: {new_var}"
        }

        self.strategies["Distractor"] = {
            "description": "Add a new, confusingly-worded but logically superfluous clue to a logic puzzle to increase cognitive load.",
            "variable_extractor": lambda p: "[PLACEHOLDER_FOR_A_NEW_CONFUSING_CLUE]",
            "loss_instruction_template": (
                "The following is a math problem, a model's correct reasoning, and the specific distractor text it was given. "
            "The model successfully ignored the distractor. Your task is to critique the *distractor text*. "
            "Explain why it was not confusing enough and suggest how it could be made more salient or thematically integrated to trick the model into using it. "
            "Be very specific in your feedback on the distractor."
            "\n\n--- MATH PROBLEM ---\n{problem}\nAnswer should be {correct_answer}."
            "\n\n--- MODEL'S CORRECT REASONING ---\n{successful_trace}"
            ),
            "reconstructor": lambda p, new_var: f"{p['full_problem_text']}\n- New Clue: {new_var}"
        }
        
        print(f"Initialized with {len(self.strategies)} adversarial strategies.")

    def harden_problem(self, problem, successful_trace, distractor, correct_answer):
        """
        Orchestrates the full hardening process for a single problem.

        Args:
            problem (dict): The problem dictionary, which must contain the necessary keys
                            for the chosen strategy (e.g., 'framing', 'question').
            successful_trace (str): The Proposer's correct reasoning trace.

        Returns:
            A string of the new, hardened problem, or None if the process fails.
        """
        # --- 1. Meta-Cognitive Strategy Selection ---
        # strategy_descriptions = "\n".join([f"- **{name}**: {config['description']}" for name, config in self.strategies.items()])
        # selector_prompt = (
        #     f"A student model correctly solved this problem:\n---PROBLEM---\n{problem['question']}\n\n"
        #     f"Here was its reasoning:\n---REASONING---\n{successful_trace}\n\n"
        #     "Based on the problem structure and the model's reasoning, which of the following strategies is most likely to create a new version of this problem that stumps the model?\n"
        #     f"{strategy_descriptions}\n\n"
        #     "Respond in JSON format: {\"strategy\": \"Chosen Strategy Name\", \"rationale\": \"Your brief rationale for this choice...\"}"
        # )
        
        # response_str = self.selector.invoke(selector_prompt, temperature=0.2)
        # try:
        #     choice = json.loads(response_str)
        #     strategy_name = choice['strategy']
        #     rationale = choice.get('rationale', '')
        #     if strategy_name not in self.strategies:
        #         raise ValueError("Selector chose an unknown strategy.")
        # except (json.JSONDecodeError, TypeError, ValueError) as e:
        #     print(f"Selector failed to choose a valid strategy: {e}")
        #     return None
        
        # print(f"AACE selected strategy: {strategy_name}")

        strategy_name = "Distractor"

        # --- 2. Execute the Chosen TextGrad Strategy ---
        strategy_config = self.strategies[strategy_name]
        
        # Extract the part of the problem to be optimized
        # original_variable_text = strategy_config["variable_extractor"](problem)
        # if original_variable_text is None:
        #     print(f"Strategy '{strategy_name}' is not applicable to this problem (variable not found).")
        #     return None

        # Create the TextGrad Variable and Loss Function
        print("Distractor: ", distractor)
        distractor = distractor
        variable = tg.Variable(distractor, requires_grad=True, role_description="A part of a problem to be adversarially rewritten.")
        context = problem.get('full_problem_text', problem['question'])
        loss_instruction = strategy_config["loss_instruction_template"].format(problem=problem, correct_answer = correct_answer, 
                                                                              successful_trace = successful_trace)
        # .format(rationale=rationale, context=context)
        print("Loss: ", loss_instruction)
        loss_fn = tg.TextLoss(loss_instruction)
        optimizer = tg.TGD(parameters=[variable], engine=self.optimizer_engine)

        # try:
        # Run the TextGrad cycle
        loss = loss_fn(variable)
        print(f"TextGrad Critique (Gradient) for f{strategy_name}: {loss.value}")
        loss.backward()
        optimizer.step()
        
        new_variable_text = variable.value
        
        # Reconstruct the full problem with the optimized part
        hardened_problem_text = strategy_config["reconstructor"](problem, new_variable_text)
        
        return [hardened_problem_text, variable.value]

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

class IterativeHardeningPipeline:
    def __init__(self, proposer_model_name: str, oracle_model_name: str, max_iterations: int = 3):
        self.proposer = HuggingFaceAgent(proposer_model_name)
        self.optimizer = AdversarialProblemOptimizer(oracle_model=oracle_model_name)
        self.validator = OpenAIAgent(oracle_model_name, "You are a validation expert. Check if a math problem is logically coherent and has one unambiguous answer. Respond with 'VALID' or 'INVALID' and a brief explanation.")
        self.max_iterations = max_iterations
        self.distractor_proposer = OpenAIAgent(oracle_model_name, "You are a math expert. Add a new, confusingly-worded but logically superfluous clue to a math problem to increase cognitive load, without changing the final answer.")

    def process_problem(self, seed_problem: dict):
        # Adapt seed problem structure for our optimizer (example for GSM8k)
        ground_truth_solution = seed_problem['solution'] 
        problem = {
            'framing': "", # GSM8k has no separate framing
            'core_facts': seed_problem['problem'],
            'question': "", # The question is part of the core_facts
            'answer': int(extract_aime_answer_from_solution_v2(ground_truth_solution)),
            'full_problem_text': seed_problem['problem']
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
            # if not verify_aime_answer(solution_trace, str(problem['answer'])):
            #     print("SUCCESS: Proposer failed! Adversarial example found.")
            #     # This is our final, high-quality data point.
                
            #     return {
            #         "seed_problem": seed_problem['problem'],
            #         "final_hardened_problem": current_prompt,
            #         "correct_answer": problem['answer'],
            #         "proposer_reasoning_traces": reasoning_traces, # Contains all attempts
            #         "proposer_answers": answers,
            #         "final_failed_trace": solution_trace,
                        # "problems": problems,
            #     }
            
            print("Proposer succeeded. Attempting to harden the problem...")
            
            if i == 0:
                distractor = self.distractor_proposer.invoke("Given the following problem, successful solution trace and answer, generate distractor text. ")
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
                f"Is the following problem valid and unambiguous? \n\n{hardened_prompt}",
                temperature=1.0
            )
            print("Validation: ", validation_response)
            if "INVALID" in validation_response.upper():
                print(f"Generated problem failed validation. Stopping iteration. Reason: {validation_response}")
                return None
            
            # Update the prompt for the next loop
            current_prompt = hardened_prompt
            problem['full_problem_text'] = hardened_prompt # Update context for next optimization
            print("Harder problem: ", hardened_prompt)
            problems.append(hardened_prompt)
        print("Max iterations reached, but proposer kept succeeding. No final failure example generated.")
        return None