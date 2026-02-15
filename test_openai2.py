from init import *

# try:
    # Make a request to the chat completions endpoint
response = client.chat.completions.create(
    # Specify the model to use, for example, "gpt-4o" or "gpt-3.5-turbo"
    model="gpt-5.2-2025-12-11", 
    
    # The 'messages' parameter is a list of message objects,
    # where each object has a 'role' and 'content'.
    # The conversation starts with a system message to set the context.
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
)

# Print the content of the assistant's response
print(response.choices[0].message.content)

# except Exception as e:
#     print(f"An error occurred: {e}")