from agents.openaiagent import OpenAI

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

    def invoke(self, user_prompt: str, temperature: float = 1) -> str | None:
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
            print(f"OpenAI API call failed for model {self.model}: {e}")
            return None
