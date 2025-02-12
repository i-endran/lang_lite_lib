from enum import Enum


class LLMProvider(Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    XAI = "xai"
    DEEPSEEK = "deepseek"

DEFAULT_MODELS = {
    "google": {
        "llm": "gemini-2.0-flash",
        "embedding": "models/embedding-001",
    },
    "openai": {
        "llm": "gpt-4o",
        "embedding": "text-embedding-3-large",
    },
    "xai": {
        "llm": "grok-beta",
    },
    "deepseek": {
        "llm": "deepseek-chat",
    },
}


def get_default_llm(llm_provider: LLMProvider):
    return DEFAULT_MODELS.get(llm_provider.value).get("llm")


def get_default_embedding(llm_provider: LLMProvider):
    return DEFAULT_MODELS.get(llm_provider.value).get("embedding")
