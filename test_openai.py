import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the sys.path to allow for local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from init import *
from agents.openaiagent import OpenAIAgent

class TestOpenAIAgent(unittest.TestCase):

    def setUp(self):
        """Set up for tests."""
        self.model_name = "gpt-5-2025-08-07"
        self.system_prompt = "You are a helpful assistant."
        # We patch the OpenAI client in the setUp to avoid real API calls during initialization
        self.agent1 = OpenAIAgent(model_name=self.model_name, system_prompt=self.system_prompt)
        with patch('agents.openaiagent.OpenAI') as MockOpenAI:
            self.agent = OpenAIAgent(model_name=self.model_name, system_prompt=self.system_prompt)
            self.mock_client = MockOpenAI.return_value
            self.agent.client = self.mock_client

    def test_initialization(self):
        """Test that the agent is initialized correctly."""
        self.assertEqual(self.agent.model, self.model_name)
        self.assertEqual(self.agent.system_prompt, self.system_prompt)
        self.assertIsNotNone(self.agent.client)

    def test_invoke_success(self):
        """Test a successful invoke call."""
        print("\n--- Running test_invoke_success ---")
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Hello, world!  "
        self.mock_client.chat.completions.create.return_value = mock_response

        user_prompt = "Say hello"
        temperature = 1
        print(f"User Prompt: '{user_prompt}', Temperature: {temperature}")
        
        response = self.agent.invoke(user_prompt, temperature=temperature)
        print(f"Mocked API Response: '{mock_response.choices[0].message.content}'")
        print(f"Actual Response from invoke: '{response}'")

        response2 = self.agent1.invoke("what is 2 + 2?")
        print("Response2: ", response2)

        print("____")

        # Assertions
        self.assertEqual(response, "Hello, world!")
        print("Assertion Passed: Response is as expected.")
        self.mock_client.chat.completions.create.assert_called_once_with(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )

    def test_invoke_api_error(self):
        """Test invoke call when API raises an exception."""
        print("\n--- Running test_invoke_api_error ---")
        # Configure the mock to raise an exception
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

        user_prompt = "This will fail"
        print(f"User Prompt: '{user_prompt}' (expecting API error)")
        
        response = self.agent.invoke(user_prompt)
        print(f"Actual Response from invoke: {response}")

        # Assertions
        self.assertIsNone(response)
        print("Assertion Passed: Response is None as expected.")
        self.mock_client.chat.completions.create.assert_called_once()

if __name__ == '__main__':
    unittest.main()
