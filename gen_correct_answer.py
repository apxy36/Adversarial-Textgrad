
from init import *
import os
import re
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm.notebook import tqdm
from openai import OpenAI
import textgrad as tg
from dataset_gen import *
# from eval.MATH_Perturb.evaluate_perturb import extract_math_answer, extract_math_perturb_ground_truth_answer,  strip_string
# from eval.MATH_Perturb.evaluate_perturb import answer_check

# --- Helper Function for Answer Verification (same as before) ---
# def xtract_aime_answer_from_solution_v2(text: str) -> str | None:
#     numbers = re.findall(r'\d+', text.replace(",", ""))
#     return numbers[-1] if numbers else None

# def verify_gsm8k_answer(generated_solution: str, correct_answer: str) -> bool:
#     extracted_ans = xtract_aime_answer_from_solution_v2(generated_solution)
#     return extracted_ans == correct_answer



# --- TextGrad Corrector for Stage 2 ---
class TextGradFinalCorrector:
    """
    Uses TextGrad to optimize a failed reasoning trace into a correct one,
    using a rich context for guidance.
    """
    def __init__(self, oracle_model: str):
        self.oracle_model_llm = ChatOpenAI(model_string=oracle_model, is_multimodal=tge._check_if_multimodal(oracle_model))
        tg.set_backward_engine(self.oracle_model_llm, override=True)
        self.optimizer_engine = ChatOpenAI(model_string=oracle_model, is_multimodal=tge._check_if_multimodal(oracle_model))
        student_prompt = (
            """You are an expert mathematician solving a problem from a math competition. Provide a rigorous, step-by-step solution to the problem.
            Be concise and efficient in your reasoning.
            You MUST format and put your final answer strictly within \\boxed{}."""
        )
        self.student_engine = UnslothAgent("Qwen/Qwen3-4B", prompt=student_prompt)

    def correct(self, hardened_problem: str, failed_trace: str, last_successful_trace: str, correct_answer: str, original_problem: str, incorrect_answer: str) -> str | None:
        """
        Performs a single, context-rich TextGrad optimization step.
        """
        # The failed trace is the variable to be optimized.
        solution_variable = tg.Variable(
            failed_trace,
            requires_grad=True,
            role_description=f"A flawed, step-by-step solution to the math problem: '{hardened_problem}'. Its final answer is incorrect, and needs to be fixed while preserving as much of the original reasoning as possible - it is to be improved, not rewritten entirely. Ensure that the final answer is strictly formatted and put within \\boxed{''}"
        )

        # The loss instruction is highly contextual, providing all necessary info for a high-quality critique.
        loss_instruction = (
            "You are a math tutor evaluating a student's incorrect solution to a mathematics problem, which has been augmented (made more difficult) from an original maths problem that has been provided. Your goal is to provide a critique that will help them fix it. "
            "Do not solve the problem yourself, only identify the primary flaw in their reasoning to solve the augmented problem correctly. Guide them towards the correct final answer by improving their existing solution, rather than rewriting it completely. "
            "Note that the answer to the augmetned problem is very likely (but not guaranteed) to be the same as the original answer."
            f"\n\n--- THE AUGMENTED PROBLEM THE STUDENT FAILED ON ---\n{hardened_problem}"
            f"\n\n--- THE STUDENT'S LAST *CORRECT* REASONING TRACE (on an easier version of the problem) ---\n{last_successful_trace}"
            f"\n\n--- THE CORRECT FINAL ANSWER TO THE ORIGINAL PROBLEM---\n{correct_answer}"
            f"\n\n--- THE STUDENT'S INCORRECT ANSWER ---\n{incorrect_answer}"
            f"\n\n--- THE ORIGINAL PROBLEM ---\n{original_problem}"
            "\n Ensure that the final answer is strictly formatted and put within \\boxed{}. DO NOT use these square brackets in any other part of the solution. Ensure that the think tokens (e.g. <think>), if any, are preserved in the corrected solution."
            "\n\nBased on all this context, provide a concise critique of the student's FAILED solution to the augmented problem."
        )
        loss_fn = tg.TextLoss(loss_instruction)
        optimizer = tg.TGD(parameters=[solution_variable], engine=self.optimizer_engine)

        try:
            # Run the TextGrad optimization cycle
            loss = loss_fn(solution_variable)
            print(f"  [TextGrad Critique (Gradient)]: {loss.value}")
            loss.backward()
            optimizer.step()
            
            return solution_variable.value
        except Exception as e:
            print(f"  ERROR: TextGrad correction step failed: {e}")
            return None
        
    def correct2(self, hardened_problem: str, failed_trace: str, last_successful_trace: str, correct_answer: str, original_problem: str, incorrect_answer: str) -> str | None:
        """
        Performs the correction attempt.
        """
        # The failed trace is the variable to be optimized.
        solution_variable = tg.Variable(
            failed_trace,
            requires_grad=True,
            role_description=f"A flawed, step-by-step solution to the math problem: '{hardened_problem}'."
        )

        # The loss instruction generates the critique
        loss_instruction = (
            "You are a math tutor evaluating a student's incorrect solution to a mathematics problem, which has been augmented (made more difficult) from an original maths problem that has been provided. Your goal is to provide a critique that will help them fix it. "
            "Do not solve the problem yourself, only identify the primary flaw in their reasoning to solve the augmented problem correctly. Guide them towards the correct final answer by improving their existing solution, rather than rewriting it completely. "
            "Note that the answer to the augmetned problem is very likely (but not guaranteed) to be the same as the original answer."
            f"\n\n--- THE AUGMENTED PROBLEM THE STUDENT FAILED ON ---\n{hardened_problem}"
            f"\n\n--- THE STUDENT'S LAST *CORRECT* REASONING TRACE (on an easier version of the problem) ---\n{last_successful_trace}"
            f"\n\n--- THE CORRECT FINAL ANSWER TO THE ORIGINAL PROBLEM---\n{correct_answer}"
            f"\n\n--- THE STUDENT'S INCORRECT ANSWER ---\n{incorrect_answer}"
            f"\n\n--- THE ORIGINAL PROBLEM ---\n{original_problem}"
            "\n Ensure that the final answer is strictly formatted and put within \\boxed{}. DO NOT use these square brackets in any other part of the solution. Ensure that the think tokens (e.g. <think>), if any, are preserved in the corrected solution."
            "\n\nBased on all this context, provide a concise critique of the student's FAILED solution to the augmented problem."
        )
        
        loss_fn = tg.TextLoss(loss_instruction)
        # Optimizer is only used if fallback is needed
        optimizer = tg.TGD(parameters=[solution_variable], engine=self.optimizer_engine)

        try:
            # 1. GENERATE CRITIQUE (Forward Pass)
            loss = loss_fn(solution_variable)
            critique = loss.value
            print(f"  [Oracle Critique Generated]: {critique}...")

            # 2. ATTEMPT STUDENT SELF-CORRECTION
            print("  [Attempting Student Self-Correction]...")

            max_tries = 3
            student_valid = False
            num_tries = 0
            while (not student_valid and num_tries < max_tries):
                student_prompt = (
                    "You previously attempted this problem and failed, and also attempted an easier version of this problem and suceeded. Please read the following information and try to solve the problem again correctly.\n\n"
                    f"--- Problem ---\n{hardened_problem}\n\n"
                    f"--- Your Previous Incorrect Attempt ---\n{failed_trace}\n\n"
                    f"--- Your Last Successful Attempt (on an easier version of the problem) ---\n{last_successful_trace}\n\n"
                    f"--- Tutor's Critique/Hint ---\n{critique}\n\n"
                    "--- Instructions ---\n"
                    "Provide a new, corrected step-by-step solution. "
                    "Ensure that you put your final answer within \\boxed{}."
                )
                
                # Call student model
                student_response = self.student_engine.solve(student_prompt)
            
                if not student_response:
                    print("  - STUDENT CORRECTION FAILED: Student model failed to produce a solution.")
                    return None
            
                if verify_gsm8k_answer(student_response, correct_answer):
                    print("  + STUDENT CORRECTION PASSED: Student's answer is correct.")
                    student_valid = True
                    return student_response
                else:
                    print("  - STUDENT CORRECTION FAILED: Student's answer is still incorrect.")
                    print(f"    - Expected Answer: {correct_answer}")
                    print(f"    - Student's Answer: {extract_aime_answer_from_solution_v2(student_response)}")
                    print(f"    - Student's Solution trace: {student_response} [END]")
                    num_tries += 1

                if num_tries >= max_tries:
                    print("  - STUDENT CORRECTION FAILED: Max attempts reached.")
                    break
            
            # student_prompt = (
            #     "You previously attempted this problem and failed. Please read the critique of your previous attempt and try to solve the problem again correctly.\n\n"
            #     f"--- Problem ---\n{hardened_problem}\n\n"
            #     f"--- Your Previous Attempt ---\n{failed_trace}\n\n"
            #     f"--- Tutor's Critique/Hint ---\n{critique}\n\n"
            #     "--- Instructions ---\n"
            #     "Provide a new, corrected step-by-step solution. "
            #     "Ensure that you put your final answer within \\boxed{}."
            # )
            
            # # Call student model
            # student_response = self.student_engine.generate(student_prompt)
            
            # # Verify Student Response
            # if verify_gsm8k_answer(student_response, correct_answer):
            #     print("  [SUCCESS] Student successfully self-corrected!")
            #     print(f"  [Student's Corrected Solution]: {student_response[:100]}...")
            #     return student_response
            # else:
            #     print("  [FAILURE] Student failed to self-correct. Proceeding to Oracle Fallback...")
            #     print(f"  [Student's Incorrect Solution]: {student_response[:100]}...")

            # 3. FALLBACK: ORACLE OPTIMIZATION (Backward Pass + Step)
            # If student failed, we let the Oracle write the solution using the TGD optimizer.
            # We might update the role description for the optimizer to ensure it writes the solution directly now.
            solution_variable.set_role_description("A corrected, perfect solution to the math problem.")
            
            loss.backward() # Propagate the critique
            optimizer.step() # Oracle rewrites the variable
            
            print(f"  [Oracle Correction Completed]")
            return solution_variable.value

        except Exception as e:
            print(f"  ERROR: TextGrad correction step failed: {e}")
            return None
# 4808 + 240
# --- The Main Stage 2 Pipeline Class ---
class CorrectionVerificationPipeline:
    def __init__(self, oracle_model_name: str):
        self.corrector = TextGradFinalCorrector(oracle_model=oracle_model_name)
        # The Verifier is "blind" - it doesn't know the answer beforehand.
        self.verifier = OpenAIAgent(
            oracle_model_name,
            "You are an expert mathematician. Solve the following problem from scratch, showing all your work. Be rigorous and accurate. Strictly format your answer in this format: [[your answer here]]"
        )

    def process_hardened_data(self, hardened_data: dict):
        """
        Processes a single JSON object from the Stage 1 output file.
        """
        hardened_problem = hardened_data['final_hardened_problem']
        correct_answer = hardened_data['correct_answer']
        
        # The Proposer's final, failed attempt will be our `y_rejected`.
        y_rejected = hardened_data['final_failed_trace']
        original_problem = hardened_data['seed_problem']
        incorrect_answer = hardened_data['proposer_answers'][-1]
        original_reasoning_trace = hardened_data['proposer_reasoning_traces'][0]['trace']
        # Get the last successful trace from the list (it's the second to last one).
        # This provides context for how the model was reasoning *before* it failed.
        if len(hardened_data['proposer_reasoning_traces']) > 1:
            last_successful_trace = hardened_data['proposer_reasoning_traces'][-2]['trace']
            
        else:
            # This can happen if the model failed on the very first attempt in Stage 1.
            last_successful_trace = "The model had no prior successful attempts on this problem."

        print(f"\nProcessing problem...")
        print(f"  Hardened Problem: {hardened_problem[:100]}...")

        # --- 1. Verifier Stage ---
        print("  Running blind verification...")
        verifier_prompt = (
            "The following is an augmented math problem, which has been made more challenging from an original problem. They are intended to have the same answer, although this is not guaranteed. The reasoning trace provided below is a solution to the original problem. "
            f"\n\n Augmented Problem: {hardened_problem}"
            f"\n\n Original Problem: {original_problem}"
            f"\n\n Original Reasoning Trace: {original_reasoning_trace}"
            "\n\n Ensure that the final answer is STRICTLY formatted and put within \\boxed{}. Any other formatting is invalid."
            "\n\n Using only the information above, solve the augmented problem step-by-step."
        )

        max_tries = 3
        verifier_valid = False
        num_tries = 0
        while (not verifier_valid and num_tries < max_tries):
            verifier_solution = self.verifier.invoke(verifier_prompt, temperature=1.0)

            print(f"  Verifier's Solution Attempt: {verifier_solution}...")
            
            if not verifier_solution:
                print("  - VERIFICATION FAILED: Oracle Verifier failed to produce a solution.")
                return None
            
            if not verify_gsm8k_answer(verifier_solution, correct_answer):
                print("  - VERIFICATION FAILED: Oracle Verifier's answer does not match the ground truth.")
                print(f"    - Expected Answer: {correct_answer}")
                print(f"    - Verifier's Answer: {extract_aime_answer_from_solution_v2(verifier_solution)}")
                print(f"    - Verifier's Solution trace: {verifier_solution} [END]")
                print(f"    - Problem: {hardened_problem} [END]")
                # This is a critical failure, indicating the hardened problem is likely invalid.
                # return None
                num_tries += 1
            else:
                verifier_valid = True

            if num_tries >= max_tries:
                return None
        print("  + VERIFICATION PASSED: Blind oracle's answer is consistent with the ground truth.")

        # --- 2. TextGrad Correction Stage ---
        print("  Optimizing the failed solution using TextGrad...")
        # Use TextGrad to turn the failed trace (`y_rejected`) into a correct one (`y_chosen`).
        max_tries = 3
        textgrad_valid = False
        num_tries = 0
        while (not textgrad_valid and num_tries < max_tries):
            y_chosen = self.corrector.correct2(
                hardened_problem=hardened_problem,
                failed_trace=y_rejected,
                last_successful_trace=last_successful_trace,
                correct_answer=correct_answer,
                original_problem=original_problem,
                incorrect_answer=incorrect_answer
            )
        
            if not y_chosen:
                print("  - CORRECTION FAILED: TextGrad could not generate a corrected solution.")
                return None
            
            print(f'y chosen: {y_chosen}')
            
            # --- 3. Final Check ---
            # A final sanity check to ensure the TextGrad-corrected answer is actually correct.
            if verify_gsm8k_answer(y_chosen, correct_answer):
                print("  + CORRECTION SUCCESSFUL: Final preference pair generated.")
                
                
                textgrad_valid = True
                return {
                    "prompt": hardened_problem,
                    "chosen": y_chosen,
                    "rejected": y_rejected
                }
            else:
                print("  - CORRECTION FAILED: TextGrad-optimized solution is still incorrect.")
                print(f"correct answer: {correct_answer}" )
                # return None
                num_tries += 1
            if num_tries >= max_tries:
                return None
        return None

# --- Main Execution Block for Kaggle Notebook ---

# Configuration
ORACLE_MODEL_NAME = "gpt-5.2-2025-12-11"
# "gpt-5-2025-08-07" #"gpt-4.1"
# Input file from the Stage 1 pipeline
INPUT_FILE = "final_preference_pairs.jsonl" 
# The final output file for fine-tuning
OUTPUT_FILE = "SFTpairs6.jsonl"

# Initialize the Stage 2 pipeline
pipeline_stage2 = CorrectionVerificationPipeline(oracle_model_name=ORACLE_MODEL_NAME)

# Load the hardened data from Stage 1
try:
    with open(INPUT_FILE, 'r') as f:
        hardened_data_list = [json.loads(line) for line in f]
        print(f"Loaded {len(hardened_data_list)} hardened problems from Stage 1 output.")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE}. Please run the Stage 1 pipeline first.")
    # In a notebook, you might want to stop execution here.
    hardened_data_list = []

# Run the pipeline
successful_pairs = 0
if hardened_data_list:
    with open(OUTPUT_FILE, 'w') as f:
        for hardened_data in tqdm(hardened_data_list, desc="Correcting and Verifying Problems"):
            result = pipeline_stage2.process_hardened_data(hardened_data)
            if result:
                successful_pairs += 1
                print("CORRECTION COUNT: ", successful_pairs)
                f.write(json.dumps(result) + '\n')
                

    print(f"\n--- PIPELINE STAGE 2 COMPLETE ---")
    print(f"Successfully generated and verified {successful_pairs} final preference pairs.")
    print(f"Output saved to: {OUTPUT_FILE}")
else:
    print("No hardened data to process.")