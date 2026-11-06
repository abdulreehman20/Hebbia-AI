"""OpenRouter chat model factory for planning, tools, and generation."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import Settings, get_settings


def get_chat_model(
    settings: Settings | None = None,
    *,
    temperature: float | None = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """Return an OpenAI-compatible ChatOpenAI pointed at OpenRouter."""
    settings = settings or get_settings()
    settings.require_openrouter()
    return ChatOpenAI(
        model=settings.openrouter_chat_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.chat_temperature if temperature is None else temperature,
        max_tokens=settings.chat_max_tokens,
        streaming=streaming,
        timeout=settings.request_timeout_seconds,
        max_retries=min(settings.max_retries, 3),
        default_headers={
            "HTTP-Referer": "https://github.com/hebbia/backend",
            "X-Title": "Financial Research Agentic RAG",
        },
    )
