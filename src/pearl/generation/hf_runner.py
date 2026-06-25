"""Hugging Face Transformers generation runner."""

from __future__ import annotations

from pearl.generation.base_llm import BaseLLM, GenerationConfig


class HuggingFaceRunner(BaseLLM):
    """Lazy HF runner.

    The class is intentionally lightweight until instantiated with installed
    dependencies and local model access.
    """

    def __init__(self, model_id: str, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers to use HuggingFaceRunner") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            do_sample=config.temperature > 0,
            temperature=max(config.temperature, 1e-6),
            max_new_tokens=config.max_new_tokens,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
