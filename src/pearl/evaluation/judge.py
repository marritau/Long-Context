"""LLM-as-judge placeholder."""

from __future__ import annotations


def judge_answer_supported(*args, **kwargs) -> float:
    """Placeholder for optional LLM-as-judge evaluation."""

    raise NotImplementedError("Wire an API or local judge model before using judge metrics.")
