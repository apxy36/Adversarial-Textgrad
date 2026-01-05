from agents.openaiagent import OpenAIAgent
import textgrad as tg
import textgrad.engine as tge
from textgrad.engine.openai import ChatOpenAI
import json
import logging

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
            "Be very specific in your feedback on the distractor. Ensure that the distractor text is unambiguous and does not change the final answer of the problem with the distractor added."
            "Also, examine if the distractor makes the problem invalid or changes its answer; if so, point that out. It must not do that. "
            "Ensure that the distractor is concise and short (2 sentences or less)."
            "\n\n--- MATH PROBLEM ---\n{problem}\nAnswer should be {correct_answer}."
            "\n\n--- MODEL'S CORRECT REASONING ---\n{successful_trace}"
            ),
            "reconstructor": lambda p, new_var: f"{p['full_problem_text']}\n- Info: {new_var}"
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
        # distractor = distractor
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
            
        # except Exception as e:
        #     print(f"TextGrad optimization for strategy '{strategy_name}' failed: {e}")
        #     return None

# import textgrad as tg
# import logging
# import json

# [Assume OpenAIAgent is defined]

class OpenEndedAdversarialOptimizer:
    """
    A fully generalized optimizer that invents a custom adversarial strategy
    for each specific problem instance.
    """
    def __init__(self, oracle_model: str):
        self.oracle_model = oracle_model
        
        # The Architect invents the failure mode/attack plan
        self.architect = OpenAIAgent(
            oracle_model,
            "You are a Red Team Architect. Your goal is to analyze a specific math problem and a model's successful solution to it. "
            "You must diagnose the likely cognitive shortcuts the model took. "
            "Then, devise a NOVEL, SPECIFIC strategy to rewrite this exact problem to exploit those shortcuts and induce failure, "
            "without changing the underlying logic or the final answer."
        )
        
        # TextGrad setup
        tg.set_backward_engine(oracle_model, override=True)
        self.optimizer_engine = tg.get_engine(oracle_model)

    def harden_problem(self, problem_text: str, successful_trace: str, correct_answer: str) -> str | None:
        """
        Dynamically hardens a problem by synthesizing a custom attack strategy.
        """
        
        # --- Step 1: Synthesize the Failure Mode (The "Attack Plan") ---
        architect_prompt = (
            f"--- PROBLEM ---\n{problem_text}\n\n"
            f"--- MODEL'S SUCCESSFUL TRACE ---\n{successful_trace}\n\n"
            "Analyze the model's reasoning. Did it rely on simple keyword matching? Did it assume a standard format? "
            "Did it skip a verification step?\n\n"
            "Based on this, write a concise, imperative instruction for an optimizer to rewrite the problem. "
            "The instruction should describe EXACTLY how to mutate the text to break the model's specific reasoning path. "
            "Do not just say 'make it harder'. Be specific (e.g., 'Embed the critical variable X inside a conditional clause about Y').\n\n"
            "Output valid JSON: {\"diagnosis\": \"...\", \"attack_instruction\": \"...\"}"
        )
        
        response_str = self.architect.invoke(architect_prompt, temperature=0.7) # Higher temp for creativity
        
        try:
            plan = json.loads(response_str)
            diagnosis = plan['diagnosis']
            attack_instruction = plan['attack_instruction']
        except:
            logging.error("Architect failed to generate a valid JSON attack plan.")
            return None

        print(f"Architect Diagnosis: {diagnosis}")
        print(f"Generated Attack Instruction: {attack_instruction}")

        # --- Step 2: Configure TextGrad with the Dynamic Instruction ---
        
        # The variable is the ENTIRE problem text. 
        # We give the optimizer total freedom to rewrite it.
        problem_variable = tg.Variable(
            problem_text,
            requires_grad=True,
            role_description="A math problem to be adversarially rewritten."
        )

        # We construct the Loss Instruction dynamically using the Architect's output.
        # This is the "Placeholder Pipeline" becoming concrete.
        dynamic_loss_instruction = (
            f"You are an adversarial content generator. {attack_instruction} "
            "Ensure the final answer remains mathematically identical to the original answer. "
            "Critique the current problem text, with reference to the successful trace. If it does not yet fully implement this attack strategy, "
            "rewrite it to be more challenging to solve."
            f"Original Answer: {correct_answer}"
            f"Successful Trace: {successful_trace}"
            # f"\n\n--- CURRENT PROBLEM TEXT ---\n{input}"
        )
        
        loss_fn = tg.TextLoss(instruction=dynamic_loss_instruction)
        optimizer = tg.TGD(parameters=[problem_variable], engine=self.optimizer_engine)

        # --- Step 3: Optimization Loop ---
        try:
            loss = loss_fn(problem_variable)
            # The gradient here describes how well the current text matches the *dynamic* attack plan
            print(f"TextGrad Dynamic Critique: {loss.value}") 
            loss.backward()
            optimizer.step()
            
            return problem_variable.value
            
        except Exception as e:
            logging.error(f"TextGrad dynamic optimization failed: {e}")
            return None