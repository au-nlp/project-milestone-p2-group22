from abc import ABC, abstractmethod


class LLMChatter(ABC):
    """Abstract base class for LLM chat models."""

    @abstractmethod
    def chat(self, model: str, messages: list[dict[str, str]]) -> dict:
        """Send messages to the LLM and return the response."""
        pass


class OllamaChatter(LLMChatter):
    """Ollama LLM chatter implementation."""

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat(model=self.model_name, messages=messages)
        return response["message"]["content"]


class AzureOpenAIChatter(LLMChatter):
    """Azure OpenAI LLM chatter implementation."""

    def __init__(self, client, deployment_name: str):
        self.client = client
        self.deployment_name = deployment_name

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.deployment(
            deployment_name=self.deployment_name, messages=messages
        )
        return response["message"]["content"]


class LLMChat:
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

        # Get model's reply
        reply = response["message"]["content"]

        # Add assistant's reply to history
        self.add_message("assistant", reply)

        return reply

    def reset(self):
        """Clear the conversation history."""
        self.history = []

    def get_history(self) -> list[dict[str, str]]:
        """Return the current message history."""
        return self.history.copy()


class CachedLLMChat(LLMChat):
    """A chat wrapper that caches responses from the LLM."""

    def __init__(self, llm: LLMChat):
        super().__init__(llm)
        self.cache = {}

    def chat(self, user_message: str) -> str:
        """Return cached response if available, else query the LLM and cache the response."""
        if user_message in self.cache:
            return self.cache[user_message]
        response = self.llm.chat(user_message)
        self.cache[user_message] = response
        return response
