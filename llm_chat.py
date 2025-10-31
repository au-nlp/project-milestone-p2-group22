from abc import ABC, abstractmethod

import ollama
from openai import AzureOpenAI

from dotenv import Dotenv


class LLMChatter(ABC):
    """Abstract base class for LLM chat models."""

    @abstractmethod
    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        """Send messages to the LLM and return the response."""
        pass


class OllamaChatter(LLMChatter):
    """Ollama LLM chatter implementation."""

    def __init__(self, model_name: str, host: str = "localhost:11434"):
        self.client = ollama.Client(host=host)
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat(model=self.model_name, messages=messages)
        return response["message"]["content"]


class AzureOpenAIChatter(LLMChatter):
    """Azure OpenAI LLM chatter implementation."""

    def __init__(self, deployment_name: str = "gpt-5-nano"):
        env = Dotenv(".env")
        AZURE_KEY = env.get("AZURE_KEY")
        AZURE_ENDPOINT = env.get("AZURE_ENDPOINT")
        api_version = "2024-12-01-preview"

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
        )

        self.deployment_name = deployment_name

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_completion_tokens=16384,
        )

        answer = response.choices[0].message.content
        return answer


class LLMChatInterface(ABC):
    """Abstract base class for LLM chat interfaces."""

    @abstractmethod
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        pass

    @abstractmethod
    def chat(self, user_message: str) -> str:
        """Send a user message to the LLM and return the response."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the conversation history."""
        pass

    @abstractmethod
    def get_history(self) -> list[dict[str, str]]:
        """Return the current message history."""
        pass


class LLMChat(LLMChatInterface):
    """LLM chat that can use Ollama or Azure OpenAI."""

    def __init__(self, llm: LLMChatter):
        """
        Initialize the chat with a specific LLM model.
        """
        self.llm = llm
        self.history: list[dict[str, str]] = []

    def __str__(self):
        lines = [f"{msg['role'].upper()}: {msg['content']}\n" for msg in self.history]
        return "\n".join(lines)

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
        response = self.llm.chat(messages=self.history)

        # Add assistant's reply to history
        self.add_message("assistant", response)

        return response

    def reset(self):
        """Clear the conversation history."""
        self.history = []

    def get_history(self) -> list[dict[str, str]]:
        """Return the current message history."""
        return self.history.copy()


class CachedLLMChat(LLMChatInterface):
    """A chat wrapper that caches responses from the LLM."""

    def __init__(self, llm: LLMChat, path: str = None):
        """Initialize the cached chat with a specific LLM chat instance.

        Args:
            llm (LLMChat): The LLM chat instance to wrap.
            path (str, optional): Path to save/load the cache. Defaults to None.
                                  Then it only caches in memory and for the session."""
        self.llm = llm
        self.cache = {}
        self.path = path
        self.load_cache()

    def load_cache(self):
        """Load the cache from the specified path."""
        if self.path:
            try:
                with open(self.path, "rb") as f:
                    import pickle

                    self.cache = pickle.load(f)
            except FileNotFoundError:
                self.cache = {}

    def save_cache(self):
        """Save the cache to the specified path."""
        with open(self.path, "wb") as f:
            import pickle

            pickle.dump(self.cache, f)

    def __str__(self):
        return self.llm.__str__()

    def add_message(self, role: str, content: str):
        self.llm.add_message(role, content)

    def chat(self, user_message: str) -> str:
        if user_message in self.cache:
            return self.cache[user_message]
        response = self.llm.chat(user_message)
        self.cache[user_message] = response
        self.save_cache()
        return response

    def reset(self):
        self.llm.reset()

    def get_history(self) -> list[dict[str, str]]:
        return self.llm.get_history()


if __name__ == "__main__":
    # Example usage with Ollama
    ollama_chatter = OllamaChatter(model_name="gemma3:4b")
    chat = LLMChat(ollama_chatter)
    print("Asking Ollama: Hello, how are you?")
    response = chat.chat("Hello, how are you?")
    print("Ollama response:", response)

    # Example usage with Azure OpenAI with caching
    azure_chatter = AzureOpenAIChatter(deployment_name="gpt-5-nano")
    chat_azure = LLMChat(azure_chatter)
    cached_chat = CachedLLMChat(chat_azure, "data/azure_cache.pkl")
    # Time to see caching in action
    print("\n\nAsking Azure OpenAI the same question twice to see caching in action:")
    import time

    start_time = time.time()
    response_azure = cached_chat.chat("Hello, how are you?")
    time_spent = time.time() - start_time
    print("Azure OpenAI response:", response_azure)
    print(f"Took {time_spent:.2f} seconds")

    print("\n\nAsking again to hit the cache:")
    start_time = time.time()
    response_azure = cached_chat.chat("Hello, how are you?")
    time_spent = time.time() - start_time
    print("Azure OpenAI response:", response_azure)
    print(f"Took {time_spent:.2f} seconds")
