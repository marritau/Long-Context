from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from bikv.packing import FixedBlockPacker, PackedSample


def normalize_answer(value: str) -> str:
    return "".join(character.lower() for character in value.strip() if character.isalnum())


@dataclass
class ModelResult:
    gold_mean_logprob: float | None
    generated_answer: str | None
    is_correct: bool | None
    elapsed_seconds: float
    peak_vram_bytes: int
    cache_lengths: list[int]
    document_cache_lengths: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_mean_logprob": self.gold_mean_logprob,
            "generated_answer": self.generated_answer,
            "is_correct": self.is_correct,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
            "cache_lengths": self.cache_lengths,
            "document_cache_lengths": self.document_cache_lengths,
        }


class KVModelBackend:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        block_size: int,
        use_chat_template: bool = True,
        prefill_chunk_size: int = 2048,
        prefill_chunk_threshold: int = 4096,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.packer = FixedBlockPacker(tokenizer, block_size, use_chat_template=use_chat_template)
        self.device = next(model.parameters()).device
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.prefill_chunk_threshold = int(prefill_chunk_threshold)

    @classmethod
    def load(
        cls,
        model_name: str,
        block_size: int,
        attention_implementation: str = "sdpa",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        prefill_chunk_size: int = 2048,
        prefill_chunk_threshold: int = 4096,
    ) -> "KVModelBackend":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for model experiments; run data preparation and tests on CPU")
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": 0},
            attn_implementation=attention_implementation,
            trust_remote_code=trust_remote_code,
        )
        return cls(
            model,
            tokenizer,
            block_size,
            use_chat_template=use_chat_template,
            prefill_chunk_size=prefill_chunk_size,
            prefill_chunk_threshold=prefill_chunk_threshold,
        )

    def pack(self, sample, **overrides) -> PackedSample:
        return self.packer.pack(sample, **overrides)

    def _start_measurement(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def _finish_measurement(self, started: float) -> tuple[float, int]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
        else:
            peak = 0
        return time.perf_counter() - started, peak

    def _prefill_document(self, packed: PackedSample, press=None) -> tuple[DynamicCache, list[int]]:
        cache = DynamicCache()
        context_ids = packed.context_ids.to(self.device)
        post_cache_compression = getattr(press, "compress_cache_after_prefill", None)
        chunkable = press is None or callable(post_cache_compression)
        use_chunks = chunkable and context_ids.shape[1] > self.prefill_chunk_threshold
        if use_chunks:
            with torch.inference_mode():
                for start in range(0, context_ids.shape[1], self.prefill_chunk_size):
                    end = min(context_ids.shape[1], start + self.prefill_chunk_size)
                    position_ids = torch.arange(start, end, device=self.device).unsqueeze(0)
                    self.model.model(
                        input_ids=context_ids[:, start:end],
                        past_key_values=cache,
                        position_ids=position_ids,
                        use_cache=True,
                    )
            if callable(post_cache_compression):
                post_cache_compression(cache)
            lengths = [int(cache.get_seq_length(layer_idx)) for layer_idx in range(len(cache))]
            return cache, lengths
        manager = press(self.model) if press is not None else _NullContext()
        with torch.inference_mode(), manager:
            self.model.model(input_ids=context_ids, past_key_values=cache, use_cache=True)
        lengths = [int(cache.get_seq_length(layer_idx)) for layer_idx in range(len(cache))]
        return cache, lengths

    def _read_query(self, packed: PackedSample, cache: DynamicCache) -> list[int]:
        """Read the question for the first time against the already-compressed document cache."""
        if packed.query_ids.shape[1] == 0:
            return [int(cache.get_seq_length(layer_idx)) for layer_idx in range(len(cache))]
        start = packed.layout.original_length
        query_ids = packed.query_ids.to(self.device)
        position_ids = torch.arange(start, start + query_ids.shape[1], device=self.device).unsqueeze(0)
        with torch.inference_mode():
            self.model.model(
                input_ids=query_ids,
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
            )
        return [int(cache.get_seq_length(layer_idx)) for layer_idx in range(len(cache))]

    def _prepare_cache(self, packed: PackedSample, press=None) -> tuple[DynamicCache, list[int], list[int]]:
        cache, document_lengths = self._prefill_document(packed, press)
        final_lengths = self._read_query(packed, cache)
        return cache, document_lengths, final_lengths

    def _prepare_query_aware_cache(
        self, packed: PackedSample, press
    ) -> tuple[DynamicCache, list[int], list[int]]:
        """Compress document and query together, matching KVPress query-aware evaluation."""
        combined_ids = torch.cat([packed.context_ids, packed.query_ids], dim=1).to(self.device)
        cache = DynamicCache()
        chunkable = bool(getattr(press, "supports_chunked_query_aware", False))
        use_chunks = chunkable and combined_ids.shape[1] > self.prefill_chunk_threshold
        if use_chunks:
            observation = int(getattr(press, "window_size"))
            split = int(combined_ids.shape[1]) - observation
            with torch.inference_mode():
                for start in range(0, split, self.prefill_chunk_size):
                    end = min(split, start + self.prefill_chunk_size)
                    position_ids = torch.arange(start, end, device=self.device).unsqueeze(0)
                    self.model.model(
                        input_ids=combined_ids[:, start:end],
                        past_key_values=cache,
                        position_ids=position_ids,
                        use_cache=True,
                    )
                position_ids = torch.arange(split, combined_ids.shape[1], device=self.device).unsqueeze(0)
                with press.force_final_observation_compression(self.model):
                    self.model.model(
                        input_ids=combined_ids[:, split:],
                        past_key_values=cache,
                        position_ids=position_ids,
                        use_cache=True,
                    )
        else:
            manager = press(self.model)
            with torch.inference_mode(), manager:
                self.model.model(input_ids=combined_ids, past_key_values=cache, use_cache=True)
        lengths = [int(cache.get_seq_length(layer_idx)) for layer_idx in range(len(cache))]
        # There is no document-only cache in this one-pass protocol. Keeping a
        # separate field prevents it from being confused with BIKV's two-pass cache.
        return cache, [], lengths

    def _score_answer_ids(self, packed: PackedSample, cache: DynamicCache, answer_ids: list[int]) -> float:
        answer = torch.tensor([answer_ids], dtype=torch.long, device=self.device)
        teacher_input = torch.cat([packed.prompt_tail_ids.to(self.device), answer[:, :-1]], dim=1)
        targets = answer
        start_position = packed.layout.original_length + int(packed.query_ids.shape[1])
        position_ids = torch.arange(
            start_position, start_position + teacher_input.shape[1], device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=teacher_input,
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
            )
            log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.mean().item())

    def score(self, packed: PackedSample, press=None) -> ModelResult:
        started = self._start_measurement()
        cache, document_lengths, cache_lengths = self._prepare_cache(packed, press)
        mean_logprob = self._score_answer_ids(packed, cache, packed.answer_ids[0].tolist())
        elapsed, peak = self._finish_measurement(started)
        return ModelResult(mean_logprob, None, None, elapsed, peak, cache_lengths, document_lengths)

    def score_candidates(self, packed: PackedSample, candidates: list[str], press=None) -> dict[str, Any]:
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("Candidate answers must be a non-empty unique list")
        started = self._start_measurement()
        cache, document_lengths, cache_lengths = self._prepare_cache(packed, press)
        scores: dict[str, float] = {}
        for candidate in candidates:
            candidate_cache = copy.deepcopy(cache)
            scores[candidate] = self._score_answer_ids(
                packed,
                candidate_cache,
                self.packer.encode_answer(candidate),
            )
            del candidate_cache
        ranked = sorted(candidates, key=lambda value: scores[value], reverse=True)
        prediction = ranked[0]
        gold = packed.answer_text
        wrong_scores = [score for value, score in scores.items() if normalize_answer(value) != normalize_answer(gold)]
        gold_score = max(
            (score for value, score in scores.items() if normalize_answer(value) == normalize_answer(gold)),
            default=float("-inf"),
        )
        elapsed, peak = self._finish_measurement(started)
        return {
            "prediction": prediction,
            "is_correct": normalize_answer(prediction) in {normalize_answer(value) for value in packed.answer_aliases},
            "scores": scores,
            "ranked_candidates": ranked,
            "gold_score": gold_score,
            "gold_margin": gold_score - max(wrong_scores) if wrong_scores else float("inf"),
            "elapsed_seconds": elapsed,
            "peak_vram_bytes": peak,
            "cache_lengths": cache_lengths,
            "document_cache_lengths": document_lengths,
            "protocol": "document_compress_then_query",
        }

    def score_candidates_query_aware(
        self, packed: PackedSample, candidates: list[str], press
    ) -> dict[str, Any]:
        """Score candidates under the upstream KVPress query-aware prefill protocol."""
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("Candidate answers must be a non-empty unique list")
        started = self._start_measurement()
        cache, document_lengths, cache_lengths = self._prepare_query_aware_cache(packed, press)
        scores: dict[str, float] = {}
        for candidate in candidates:
            candidate_cache = copy.deepcopy(cache)
            scores[candidate] = self._score_answer_ids(
                packed, candidate_cache, self.packer.encode_answer(candidate)
            )
            del candidate_cache
        ranked = sorted(candidates, key=lambda value: scores[value], reverse=True)
        prediction = ranked[0]
        gold = packed.answer_text
        wrong_scores = [score for value, score in scores.items() if normalize_answer(value) != normalize_answer(gold)]
        gold_score = max(
            (score for value, score in scores.items() if normalize_answer(value) == normalize_answer(gold)),
            default=float("-inf"),
        )
        elapsed, peak = self._finish_measurement(started)
        return {
            "prediction": prediction,
            "is_correct": normalize_answer(prediction) in {normalize_answer(value) for value in packed.answer_aliases},
            "scores": scores,
            "ranked_candidates": ranked,
            "gold_score": gold_score,
            "gold_margin": gold_score - max(wrong_scores) if wrong_scores else float("inf"),
            "elapsed_seconds": elapsed,
            "peak_vram_bytes": peak,
            "cache_lengths": cache_lengths,
            "document_cache_lengths": document_lengths,
            "protocol": "query_in_compressed_prefill",
        }

    def generate(self, packed: PackedSample, press=None, max_new_tokens: int = 8) -> ModelResult:
        started = self._start_measurement()
        cache, document_lengths, cache_lengths = self._prepare_cache(packed, press)
        position = packed.layout.original_length + int(packed.query_ids.shape[1])
        current = packed.prompt_tail_ids.to(self.device)
        generated = []
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                position_ids = torch.arange(position, position + current.shape[1], device=self.device).unsqueeze(0)
                outputs = self.model(
                    input_ids=current,
                    past_key_values=cache,
                    position_ids=position_ids,
                    use_cache=True,
                    logits_to_keep=1,
                )
                token = outputs.logits[0, -1].argmax()
                generated.append(token)
                position += current.shape[1]
                if int(token.item()) in eos_ids:
                    break
                current = token.view(1, 1)
        text = self.tokenizer.decode(torch.stack(generated), skip_special_tokens=True) if generated else ""
        normalized = normalize_answer(text)
        aliases = {normalize_answer(value) for value in packed.answer_aliases}
        correct = any(alias and alias in normalized for alias in aliases)
        elapsed, peak = self._finish_measurement(started)
        return ModelResult(None, text, correct, elapsed, peak, cache_lengths, document_lengths)

    def generate_query_aware(self, packed: PackedSample, press, max_new_tokens: int = 8) -> ModelResult:
        """Generate after a one-pass query-aware compressed prefill."""
        started = self._start_measurement()
        cache, document_lengths, cache_lengths = self._prepare_query_aware_cache(packed, press)
        position = packed.layout.original_length + int(packed.query_ids.shape[1])
        current = packed.prompt_tail_ids.to(self.device)
        generated = []
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                position_ids = torch.arange(position, position + current.shape[1], device=self.device).unsqueeze(0)
                outputs = self.model(
                    input_ids=current,
                    past_key_values=cache,
                    position_ids=position_ids,
                    use_cache=True,
                    logits_to_keep=1,
                )
                token = outputs.logits[0, -1].argmax()
                generated.append(token)
                position += current.shape[1]
                if int(token.item()) in eos_ids:
                    break
                current = token.view(1, 1)
        text = self.tokenizer.decode(torch.stack(generated), skip_special_tokens=True) if generated else ""
        normalized = normalize_answer(text)
        aliases = {normalize_answer(value) for value in packed.answer_aliases}
        correct = any(alias and alias in normalized for alias in aliases)
        elapsed, peak = self._finish_measurement(started)
        return ModelResult(None, text, correct, elapsed, peak, cache_lengths, document_lengths)

    def capture_query_aware_features(self, packed: PackedSample, press) -> list[dict[str, Any]]:
        """Run a disposable diagnostic pass; its cache is never used to score an answer."""
        diagnostic_ids = torch.cat([packed.context_ids, packed.query_ids], dim=1).to(self.device)
        diagnostic_layout = copy.copy(packed.layout)
        object.__setattr__(diagnostic_layout, "suffix_length", int(packed.query_ids.shape[1]))
        object.__setattr__(diagnostic_layout, "original_length", int(diagnostic_ids.shape[1]))
        press.layout = diagnostic_layout
        cache = DynamicCache()
        manager = press(self.model)
        with torch.inference_mode(), manager:
            self.model.model(input_ids=diagnostic_ids, past_key_values=cache, use_cache=True)
        del cache
        return list(press.records)

    def capture_rich_query_aware_features(self, packed: PackedSample, press) -> dict[str, Any]:
        """Capture v11 diagnostics in a disposable full document-plus-query pass."""
        diagnostic_ids = torch.cat([packed.context_ids, packed.query_ids], dim=1).to(self.device)
        diagnostic_layout = copy.copy(packed.layout)
        object.__setattr__(diagnostic_layout, "suffix_length", int(packed.query_ids.shape[1]))
        object.__setattr__(diagnostic_layout, "original_length", int(diagnostic_ids.shape[1]))
        press.layout = diagnostic_layout
        if diagnostic_ids.shape[1] > self.prefill_chunk_threshold:
            return self._capture_rich_query_aware_features_chunked(
                packed, press, diagnostic_ids, diagnostic_layout
            )
        cache = DynamicCache()
        started = self._start_measurement()
        manager = press(self.model)
        with torch.inference_mode(), manager:
            outputs = self.model.model(input_ids=diagnostic_ids, past_key_values=cache, use_cache=True)
            surprise = []
            for boundary in range(diagnostic_layout.n_blocks - 1):
                token_position = diagnostic_layout.data_start + (boundary + 1) * diagnostic_layout.block_size
                content_length = int(diagnostic_layout.block_content_lengths[boundary + 1])
                width = min(8, content_length)
                logits = self.model.lm_head(
                    outputs.last_hidden_state[:, token_position - 1 : token_position + width - 1]
                ).float()
                targets = diagnostic_ids[:, token_position : token_position + width]
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
                )
                surprise.append(
                    {
                        "first": float(losses[0].item()),
                        "mean_first_8": float(losses.mean().item()),
                        "max_first_8": float(losses.max().item()),
                    }
                )
        elapsed, peak = self._finish_measurement(started)
        del cache, outputs
        return {
            "layers": list(press.records),
            "boundary_surprise": surprise,
            "elapsed_seconds": elapsed,
            "peak_vram_bytes": peak,
        }

    def _capture_rich_query_aware_features_chunked(
        self, packed: PackedSample, press, diagnostic_ids: torch.Tensor, diagnostic_layout
    ) -> dict[str, Any]:
        from bikv.presses import ChunkedRichFeatureAccumulator

        cache = DynamicCache()
        started = self._start_measurement()
        accumulator = ChunkedRichFeatureAccumulator(diagnostic_layout)
        hidden_chunks = []

        def run_chunk(start: int, end: int, phase: str) -> None:
            if end <= start:
                return
            accumulator.set_chunk(phase, start)
            position_ids = torch.arange(start, end, device=self.device).unsqueeze(0)
            outputs = self.model.model(
                input_ids=diagnostic_ids[:, start:end],
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
            )
            hidden_chunks.append(outputs.last_hidden_state.detach().to("cpu"))

        blocks_per_chunk = max(1, self.prefill_chunk_size // diagnostic_layout.block_size)
        with torch.inference_mode(), accumulator(self.model):
            run_chunk(0, diagnostic_layout.data_start, "prefix")
            for first_block in range(0, diagnostic_layout.n_blocks, blocks_per_chunk):
                last_block = min(diagnostic_layout.n_blocks, first_block + blocks_per_chunk)
                start = diagnostic_layout.data_start + first_block * diagnostic_layout.block_size
                end = diagnostic_layout.data_start + last_block * diagnostic_layout.block_size
                run_chunk(start, end, "blocks")
            query_start = int(packed.context_ids.shape[1])
            run_chunk(diagnostic_layout.data_end, query_start, "context_tail")
            run_chunk(query_start, int(diagnostic_ids.shape[1]), "query")

        all_hidden = torch.cat(hidden_chunks, dim=1)
        surprise = []
        with torch.inference_mode():
            for boundary in range(diagnostic_layout.n_blocks - 1):
                token_position = diagnostic_layout.data_start + (boundary + 1) * diagnostic_layout.block_size
                content_length = int(diagnostic_layout.block_content_lengths[boundary + 1])
                width = min(8, content_length)
                hidden = all_hidden[:, token_position - 1 : token_position + width - 1].to(self.device)
                logits = self.model.lm_head(hidden).float()
                targets = diagnostic_ids[:, token_position : token_position + width]
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
                )
                surprise.append(
                    {
                        "first": float(losses[0].item()),
                        "mean_first_8": float(losses.mean().item()),
                        "max_first_8": float(losses.max().item()),
                    }
                )
        elapsed, peak = self._finish_measurement(started)
        press.records = accumulator.finalize()
        del cache, all_hidden, hidden_chunks
        return {
            "layers": list(press.records),
            "boundary_surprise": surprise,
            "elapsed_seconds": elapsed,
            "peak_vram_bytes": peak,
            "capture_mode": "causal_block_aligned_chunked_exact",
        }


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False
