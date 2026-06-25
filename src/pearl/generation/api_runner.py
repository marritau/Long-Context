"""External API runner placeholder."""

from __future__ import annotations

from pearl.generation.base_llm import BaseLLM, GenerationConfig


class ApiRunner(BaseLLM):
    """Placeholder for API-based generation."""

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        raise NotImplementedError("Configure an API client before using ApiRunner.")
