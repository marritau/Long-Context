"""LongLLMLingua baseline placeholder."""

from __future__ import annotations

from pearl.compression.llmlingua_wrapper import compress_with_llmlingua


def longllmlingua_pack(*args, **kwargs):
    """Run LongLLMLingua when the optional integration is configured."""

    return compress_with_llmlingua(*args, **kwargs)
