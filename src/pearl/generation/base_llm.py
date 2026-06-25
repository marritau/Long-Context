"""Base LLM runner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.0
    max_new_tokens: int = 256
    stop: list[str] | None = None


class BaseLLM(ABC):
    """Abstract generation runner."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate text for a prompt."""
