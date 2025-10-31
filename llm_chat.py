from abc import abstractmethod, ABC


class LLMChat(ABC):
    """Abstract base class for LLM chat implementations."""

    @abstractmethod
    def __init__(self, llm: "LLMChat"):
        self.llm = llm

    def chat(self, prompt: str) -> str:
        return self.llm.chat(prompt)


class CachedLLMChat(LLMChat):
    """A chat wrapper that caches responses from the LLM."""

    def __init__(self, llm: LLMChat):
        super().__init__(llm)
        self.cache = {}

    def chat(self, prompt):
        if prompt in self.cache:
            return self.cache[prompt]
        response = self.llm.chat(prompt)
        self.cache[prompt] = response
        return response
