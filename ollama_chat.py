import ollama
from typing import List, Dict


class OllamaChat:
    """
    A flexible chat wrapper for the Ollama Python API.
    Supports system, user, and assistant messages with conversation memory.
    """

    def __init__(self, model_name: str):
        """
        Initialize the chat with a specific Ollama model.
        """
        self.model_name = model_name
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """
        Add a message manually to the conversation.
        Role must be one of: 'system', 'user', or 'assistant'.
        """
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Role must be 'system', 'user', or 'assistant'")
        self.history.append({"role": role, "content": content})

    def chat(self, user_message: str) -> str:
        """
        Add a user message, send the full history to the model,
        and append the model's reply to the history.
        """
        # Add user's message
        self.add_message("user", user_message)

        # Send all messages (including system & assistant) to the model
        response = ollama.chat(model=self.model_name, messages=self.history)

        # Get model's reply
        reply = response["message"]["content"]

        # Add assistant's reply to history
        self.add_message("assistant", reply)

        return reply

    def reset(self):
        """Clear the conversation history."""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current message history."""
        return self.history.copy()
