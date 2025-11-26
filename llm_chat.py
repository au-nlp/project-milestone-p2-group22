import time
from abc import ABC, abstractmethod

import ollama
from gradient import Gradient
from openai import AzureOpenAI, OpenAI
from openai.types.shared.reasoning_effort import ReasoningEffort

from dotenv import Dotenv


class LLMChatter(ABC):
    """Abstract base class for LLM chat models."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        """Send messages to the LLM and return the response."""
        pass


def block_at_least_sec(n: int, start_time: float | None = None):
    """Block execution for at least n seconds.

    Args:
        n (int): Maximum number of seconds to block.
        start_time (float, optional): Start time to calculate elapsed time from. Defaults to None,
        which means the current time is used and it will then block for n seconds.
    """

    if start_time is None:
        time.sleep(n)
        return
    current_time = time.time()
    elapsed = current_time - start_time
    if elapsed < n:
        print(f"Rate limiting: sleeping for {n - elapsed:.0f} seconds...")
        time.sleep(n - elapsed)


class OllamaChatter(LLMChatter):
    """Ollama LLM chatter implementation."""

    def __init__(
        self, model_name: str, host: str = "localhost:11434", think: bool = False
    ):
        self.think = think
        self.client = ollama.Client(host=host)
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        response = self.client.chat(
            model=self.model_name, messages=messages, think=self.think
        )
        thoughts = None
        if self.think:
            thoughts = response["message"]["thinking"]
        return response["message"]["content"], thoughts


class AzureOpenAIChatter(LLMChatter):
    """Azure OpenAI LLM chatter implementation."""

    def __init__(self, deployment_name: str = "gpt-5-nano", rate: float = 1.0):
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
        self.previous_query_time: float = 0
        self.rate = rate

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        block_at_least_sec(int(60.0 / self.rate), self.previous_query_time)
        self.previous_query_time = time.time()  # TODO here or after the API call?
        completion_response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            max_completion_tokens=16384,
        )

        assistant_message = completion_response.choices[0].message.content
        if assistant_message is None:
            raise ValueError("No response from Azure OpenAI chat completion.")
        return assistant_message, None


class OpenAIChatter(LLMChatter):
    """Azure OpenAI LLM chatter implementation."""

    def __init__(
        self,
        deployment_name: str = "gpt-5-mini",
        effort_level: ReasoningEffort = "medium",
    ):
        env = Dotenv(".env")
        api_key = env.get("AZURE_KEY")
        endpoint = env.get("OPENAI_ENDPOINT")

        if api_key is None or endpoint is None:
            raise ValueError(
                "Azure API key or endpoint not found in .env file. "
                "Please set AZURE_KEY and OPENAI_ENDPOINT."
            )

        self.client = OpenAI(base_url=f"{endpoint}", api_key=api_key)
        self.deployment_name = deployment_name
        if effort_level not in {"low", "medium", "high"}:
            raise ValueError("effort_level must be one of: 'low', 'medium', 'high'")
        self.effort_level: ReasoningEffort = effort_level

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        response = self.client.responses.create(
            model=self.deployment_name,
            input=messages,
            reasoning={
                "effort": self.effort_level,
                "summary": "auto",  # or "detailed", "brief", etc.
            },
            max_output_tokens=16384,
        )

        assistant_message = None
        reasoning_summary = None

        # The Responses API returns a list of output blocks in response.output
        for block in response.output:
            if block.type == "message":
                # Extract the assistant text from the message block
                for content in block.content:
                    if content.type == "output_text":
                        assistant_message = content.text

            elif block.type == "reasoning":
                # Extract summary text
                if hasattr(block, "summary") and block.summary:
                    # summary is a list of summary_text items
                    texts = [
                        item.text
                        for item in block.summary
                        if item.type == "summary_text"
                    ]
                    reasoning_summary = "\n".join(texts)

        return assistant_message, reasoning_summary


class DigitalOceanChatter(LLMChatter):
    """Azure OpenAI LLM chatter implementation."""

    def __init__(
        self,
        deployment_name: str = "deepseek-r1-distill-llama-70b",
    ):
        env = Dotenv(".env")
        api_key = env.get("DO_DEEPSEEK")

        if api_key is None:
            raise ValueError(
                "DigitalOcean API key not found in .env file. Please set DO API key."
            )

        self.client = Gradient(model_access_key=api_key)
        self.deployment_name = deployment_name

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.deployment_name,
            max_tokens=16384,
        )

        assistant_message = response.choices[0].message.content
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

    def _get_cache_key(self, user_message: str) -> str:
        """Create a cache key from the current history and user message."""
        import json

        # Include the current history plus the new user message
        cache_data = {
            "history": self.base_chat.get_history(),
            "user_message": user_message,
        }
        return json.dumps(cache_data, sort_keys=True)

    def __str__(self):
        return self.base_chat.__str__()

    def add_message(self, role: str, content: str):
        self.base_chat.add_message(role, content)

    def chat(self, user_message: str) -> tuple[str, str | None]:
        cache_key = self._get_cache_key(user_message)

        if cache_key in self.response_cache:
            cached_result = self.response_cache[cache_key]
            # Add the cached messages to history
            self.base_chat.add_message("user", user_message)
            self.base_chat.add_message("assistant", cached_result[0])
            return cached_result

        assistant_response = self.base_chat.chat(user_message)
        self.response_cache[cache_key] = assistant_response
        self.save_cache()
        return assistant_response

    def reset(self):
        self.base_chat.reset()

    def get_history(self) -> list[dict[str, str]]:
        return self.base_chat.get_history()


if __name__ == "__main__":
    print("This is a manual test of the LLM chat implementations.")
    print("Make sure you have the .env file set up with your API keys and endpoints.")
    print("What do you want to try? Ollama, Azure, OpenAI, or all?")
    user_input = input("Type either 'ollama', 'azure', 'openai', 'do', or 'all': ")
    if user_input.lower() in {"ollama", "all"}:
        # Example usage with Ollama
        ollama_chatter = OllamaChatter(model_name="gemma3:4b")
        ollama_chat = LLMChat(ollama_chatter)
        print("Asking Ollama: Hello, how are you?")
        ollama_response = ollama_chat.chat("Hello, how are you?")
        print("Ollama response:", ollama_response)

    if user_input.lower() in {"azure", "all"}:
        # Example usage with Azure OpenAI with caching
        azure_chatter = AzureOpenAIChatter(deployment_name="gpt-5-nano")
        azure_chat = LLMChat(azure_chatter)
        cached_azure_chat = CachedLLMChat(azure_chat, "data/azure_cache.pkl")

        # Time to see caching in action
        print(
            "\n\nAsking Azure OpenAI the same question twice to see caching in action:"
        )
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

    if user_input.lower() in {"openai", "all"}:
        # Example usage with OpenAI
        openai_chatter = OpenAIChatter(deployment_name="gpt-5-mini")
        openai_chat = LLMChat(openai_chatter)

        response, thoughts = openai_chat.chat(
            (
                "You are a highly logical assistant. Solve the following river crossing problem step by step, "
                "explaining your reasoning at each stage. After reasoning, provide the final solution clearly.\n\n"
                "Problem:\n"
                "A farmer is on one side of a river with a wolf, a goat, and a cabbage. "
                "He has a boat that can carry only himself plus one item at a time. "
                "If left alone, the wolf will eat the goat, and the goat will eat the cabbage.\n\n"
                "Instructions:\n"
                "1. First, reason carefully about each move and explain why it is safe.\n"
                "2. Clearly indicate the state of the riverbanks after each move.\n"
                "3. Only after showing your detailed reasoning, provide the complete step-by-step solution.\n"
            )
        )
        print("OpenAI response:", response)
        print("OpenAI thoughts:", thoughts)

    if user_input.lower() in {"do", "all"}:
        # Example usage with DigitalOcean Claude
        do_chatter = DigitalOceanChatter()
        do_chat = LLMChat(do_chatter)

        response, _ = do_chat.chat(
            (
                "You are a highly logical assistant. Solve the following river crossing problem step by step, "
                "explaining your reasoning at each stage. After reasoning, provide the final solution clearly.\n\n"
                "Problem:\n"
                "A farmer is on one side of a river with a wolf, a goat, and a cabbage. "
                "He has a boat that can carry only himself plus one item at a time. "
                "If left alone, the wolf will eat the goat, and the goat will eat the cabbage.\n\n"
                "Instructions:\n"
                "1. First, reason carefully about each move and explain why it is safe.\n"
                "2. Clearly indicate the state of the riverbanks after each move.\n"
                "3. Only after showing your detailed reasoning, provide the complete step-by-step solution.\n"
            )
        )
        print("do response:", response)
