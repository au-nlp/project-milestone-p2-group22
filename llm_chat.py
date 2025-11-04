from abc import ABC, abstractmethod

import ollama
from openai import AzureOpenAI

from dotenv import Dotenv


class LLMChatter(ABC):
    """Abstract base class for LLM chat models."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        """Send messages to the LLM and return the response."""
        pass


class OllamaChatter(LLMChatter):
    """Ollama LLM chatter implementation."""

    def __init__(self, model_name: str, host: str = "localhost:11434", think: bool = False):
        self.think = think
        self.client = ollama.Client(host=host)
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        response = self.client.chat(model=self.model_name, messages=messages, think=self.think)
        thoughts = None
        if self.think:
            thoughts = response["message"]["thinking"]
        return response["message"]["content"], thoughts


class AzureOpenAIChatter(LLMChatter):
    """Azure OpenAI LLM chatter implementation."""

    def __init__(self, deployment_name: str = "gpt-5-nano"):
        env = Dotenv(".env")
        azure_api_key = env.get("AZURE_KEY")
        azure_endpoint = env.get("AZURE_ENDPOINT")
        api_version = "2024-12-01-preview"

        if azure_api_key is None or azure_endpoint is None:
            raise ValueError(
                "Azure API key or endpoint not found in .env file. "
                "Please set AZURE_KEY and AZURE_ENDPOINT."
            )

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
        )

        self.deployment_name = deployment_name

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        completion_response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_completion_tokens=16384,
        )

        assistant_message: str = completion_response.choices[0].message.content
        return assistant_message, None


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

    def __init__(self, chatter: LLMChatter):
        """
        Initialize the chat with a specific LLM chatter.
        """
        self.chatter = chatter
        self.message_history: list[dict[str, str]] = []

    def __str__(self):
        formatted_messages = [
            f"{msg['role'].upper()}: {msg['content']}\n" for msg in self.message_history
        ]
        return "\n".join(formatted_messages)

    def add_message(self, role: str, content: str):
        """
        Add a message manually to the conversation.
        Role must be one of: 'system', 'user', or 'assistant'.
        """
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Role must be 'system', 'user', or 'assistant'")
        self.message_history.append({"role": role, "content": content})

    def chat(self, user_message: str) -> tuple[str, str | None]:
        """
        Add a user message, send the full history to the model,
        and append the model's reply to the history.
        """
        # Add user's message
        self.add_message("user", user_message)

        # Send all messages (including system & assistant) to the model
        assistant_response, thoughts = self.chatter.chat(messages=self.message_history)

        # Add assistant's reply to history
        self.add_message("assistant", assistant_response)

        return assistant_response, thoughts

    def reset(self):
        """Clear the conversation history."""
        self.message_history = []

    def get_history(self) -> list[dict[str, str]]:
        """Return the current message history."""
        return self.message_history.copy()


class CachedLLMChat(LLMChatInterface):
    """A chat wrapper that caches responses from the LLM."""

    def __init__(self, base_chat: LLMChat, cache_file_path: str | None = None):
        """Initialize the cached chat with a specific LLM chat instance.

        Args:
            base_chat (LLMChat): The LLM chat instance to wrap.
            cache_file_path (str, optional): Path to save/load the cache. Defaults to 'data/conversations.pkl'.
                                             Then it only caches in memory and for the session."""
        self.base_chat = base_chat
        self.response_cache = {}
        self.cache_file_path = cache_file_path
        self.load_cache()

    def load_cache(self, path: str | None = None):
        """Load the cache from the specified path."""
        cache_file_path = path or self.cache_file_path
        if cache_file_path is not None:
            try:
                with open(cache_file_path, "rb") as cache_file:
                    import pickle

                    self.response_cache = pickle.load(cache_file)
            except FileNotFoundError:
                self.response_cache = {}

    def save_cache(self, path: str | None = None):
        """Save the cache to the specified path."""
        cache_file_path = path or self.cache_file_path
        if cache_file_path is not None:
            with open(cache_file_path, "wb") as cache_file:
                import pickle

                pickle.dump(self.response_cache, cache_file)

    def __str__(self):
        return self.base_chat.__str__()

    def add_message(self, role: str, content: str):
        self.base_chat.add_message(role, content)

    def chat(self, user_message: str) -> str:
        if user_message in self.response_cache:
            return self.response_cache[user_message]

        assistant_response = self.base_chat.chat(user_message)
        self.response_cache[user_message] = assistant_response
        self.save_cache()
        return assistant_response

    def reset(self):
        self.base_chat.reset()

    def get_history(self) -> list[dict[str, str]]:
        return self.base_chat.get_history()


if __name__ == "__main__":
    # Example usage with Ollama
    ollama_chatter = OllamaChatter(model_name="gemma3:4b")
    ollama_chat = LLMChat(ollama_chatter)
    print("Asking Ollama: Hello, how are you?")
    ollama_response = ollama_chat.chat("Hello, how are you?")
    print("Ollama response:", ollama_response)

    # Example usage with Azure OpenAI with caching
    azure_chatter = AzureOpenAIChatter(deployment_name="gpt-5-nano")
    azure_chat = LLMChat(azure_chatter)
    cached_azure_chat = CachedLLMChat(azure_chat, "data/azure_cache.pkl")

    # Time to see caching in action
    print("\n\nAsking Azure OpenAI the same question twice to see caching in action:")
    import time

    start_time = time.time()
    first_response = cached_azure_chat.chat("Hello, how are you?")
    first_duration = time.time() - start_time
    print("Azure OpenAI response:", first_response)
    print(f"Took {first_duration:.2f} seconds")

    print("\n\nAsking again to hit the cache:")
    start_time = time.time()
    cached_response = cached_azure_chat.chat("Hello, how are you?")
    cached_duration = time.time() - start_time
    print("Azure OpenAI response:", cached_response)
    print(f"Took {cached_duration:.2f} seconds")
