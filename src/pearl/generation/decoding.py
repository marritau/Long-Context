"""Decoding config helpers."""

from __future__ import annotations

from pearl.generation.base_llm import GenerationConfig


def make_generation_config(config: dict | None = None) -> GenerationConfig:
    """Convert a dictionary to GenerationConfig."""

    config = config or {}
    return GenerationConfig(
        temperature=float(config.get("temperature", 0.0)),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        stop=config.get("stop"),
    )
