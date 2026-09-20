from dotenv import load_dotenv
import os

from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

#Retrive API key from env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini model
llm = Gemini(
    model="models/gemini-3.6-flash",

)

def simple_chatbot():
    """
    A simple interactive chatbot using Gemini with LlamaIndex.
    """
    print("🦙 Simple Gemini Chatbot 🦙")
    print("Type 'exit' to end the conversation")
    print("-" * 50)

    # Initialize chat history
    messages = []

    while True:
        # Get user input
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