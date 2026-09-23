import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage

load_dotenv()
llm = Groq(model="qwen/qwen3.8-27b", api_key=os.getenv("GROQ_API_KEY"), max_tokens=500)

def simple_chatbot():
    """
    A simple interactive chatbot using Mistral via Ollama locally.
    """
    print("Mistral Chatbot (local via Ollama)")
    print("Type 'exit' to end the conversation")
    print("-" * 50)

    # Initialize chat history
    messages = []

    while True:
        user_input = input("\nYou: ")

        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nChatbot: Goodbye! Have a great day!")
            break

        # Add user message to history
        messages.append(ChatMessage(role="user", content=user_input))

        try:
            # Get response from Gemini
            response = llm.chat(messages)

            # Print the response
            print(f"\nChatbot: {response.message.content}")

            # Add assistant response to history
            messages.append(ChatMessage(role="assistant", content=response.message.content))

        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or check your API key")


# Run the chatbot
if __name__ == "__main__":
    simple_chatbot()