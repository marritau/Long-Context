"""vLLM generation runner."""

from __future__ import annotations

from pearl.generation.base_llm import BaseLLM, GenerationConfig


class VLLMRunner(BaseLLM):
    """vLLM runner wrapper."""

    def __init__(self, model_id: str, **kwargs):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("Install vllm to use VLLMRunner") from exc
        self.llm = LLM(model=model_id, **kwargs)

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            stop=config.stop,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()
