from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from bikv.schema import StorySample


@dataclass(frozen=True)
class CacheLayout:
    prefix_length: int
    data_start: int
    block_size: int
    n_blocks: int
    suffix_length: int
    original_length: int
    block_content_lengths: tuple[int, ...] = ()

    @property
    def data_end(self) -> int:
        return self.data_start + self.block_size * self.n_blocks

    @property
    def protected_tokens(self) -> int:
        return self.prefix_length + self.suffix_length

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PackedSample:
    sample: StorySample
    # Only the reusable document prefix is prefetched under a compression press.
    context_ids: torch.Tensor
    # The question is read only after the evaluated document cache has been compressed.
    query_ids: torch.Tensor
    prompt_tail_ids: torch.Tensor
    answer_ids: torch.Tensor
    answer_text: str
    answer_aliases: list[str]
    layout: CacheLayout

    @property
    def full_prompt_length(self) -> int:
        return self.layout.original_length + int(self.query_ids.shape[1]) + int(self.prompt_tail_ids.shape[1])


class FixedBlockPacker:
    DOCUMENT_MARKER = "BIKV_DOCUMENT_PAYLOAD_91F3A7"

    def __init__(self, tokenizer: Any, block_size: int, use_chat_template: bool = True):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.use_chat_template = use_chat_template
        self.filler_ids = tokenizer.encode(" Routine archival note.", add_special_tokens=False)
        if not self.filler_ids:
            raise ValueError("Tokenizer produced no filler tokens")

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _pack_block(self, text: str) -> tuple[list[int], int]:
        ids = self._encode(text + "\n")
        if len(ids) > self.block_size:
            raise ValueError(
                f"Evidence block needs {len(ids)} tokens but block_size={self.block_size}; "
                "increase block_size instead of truncating evidence"
            )
        content_length = len(ids)
        while len(ids) < self.block_size:
            ids.extend(self.filler_ids)
        return ids[: self.block_size], content_length

    @staticmethod
    def _find_subsequence(values: list[int], needle: list[int]) -> int:
        matches = [
            index
            for index in range(len(values) - len(needle) + 1)
            if values[index : index + len(needle)] == needle
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one document marker in chat template, found {len(matches)}")
        return matches[0]

    def _chat_shell(
        self,
        question: str,
        system_instruction: str | None = None,
        answer_instruction: str | None = None,
    ) -> tuple[list[int], list[int]]:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer has no chat template support; set use_chat_template=false only for tests")
        system = system_instruction or (
            "You solve evidence-chain questions over investigation records. "
            "Use the document facts, perform every required relation hop, and return only the requested short answer."
        )
        response_instruction = answer_instruction or "Return only the answer, without explanation."
        user = (
            f"DOCUMENT:\n{self.DOCUMENT_MARKER}\n\n"
            f"QUESTION: {question}\n"
            f"{response_instruction}"
        )
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_ids = self._encode(str(rendered))
        marker_ids = self._encode(self.DOCUMENT_MARKER)
        marker_start = self._find_subsequence(full_ids, marker_ids)
        return full_ids[:marker_start], full_ids[marker_start + len(marker_ids) :]

    def encode_answer(self, answer: str) -> list[int]:
        prefix = "" if self.use_chat_template else " "
        answer_ids = self._encode(prefix + answer)
        if not answer_ids:
            raise ValueError("Answer tokenization is empty")
        return answer_ids

    def pack(
        self,
        sample: StorySample,
        *,
        question: str | None = None,
        answer: str | None = None,
        answer_aliases: list[str] | None = None,
        system_instruction: str | None = None,
        answer_instruction: str | None = None,
    ) -> PackedSample:
        selected_question = question or sample.question
        selected_answer = answer or sample.answer
        selected_aliases = (
            list(answer_aliases)
            if answer_aliases is not None
            else list(sample.answer_aliases) if answer is None else [selected_answer]
        )
        if self.use_chat_template:
            prefix_ids, suffix_ids = self._chat_shell(
                selected_question,
                system_instruction=system_instruction,
                answer_instruction=answer_instruction,
            )
        else:
            prefix = (
                "You are reading an investigation record. Use all relevant entries and answer with only the requested "
                "short code.\n\nDOCUMENT:\n"
            )
            suffix = f"\nQUESTION: {selected_question}\nSHORT ANSWER:"
            prefix_ids = self._encode(prefix)
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_token_id is not None and (not prefix_ids or prefix_ids[0] != bos_token_id):
                prefix_ids = [int(bos_token_id)] + prefix_ids
            suffix_ids = self._encode(suffix)

        document_ids = []
        block_content_lengths = []
        token_blocks = sample.metadata.get("block_token_ids")
        if token_blocks is not None:
            if len(token_blocks) != len(sample.blocks):
                raise ValueError("Pretokenized block count does not match sample.blocks")
            for values in token_blocks:
                ids = [int(value) for value in values]
                if not ids or len(ids) > self.block_size:
                    raise ValueError("Pretokenized blocks must contain 1..block_size tokens")
                content_length = len(ids)
                while len(ids) < self.block_size:
                    ids.extend(self.filler_ids)
                document_ids.extend(ids[: self.block_size])
                block_content_lengths.append(content_length)
        else:
            for block in sample.blocks:
                packed_block, content_length = self._pack_block(block)
                document_ids.extend(packed_block)
                block_content_lengths.append(content_length)

        if not suffix_ids:
            raise ValueError("Question suffix must contain at least one token")
        query_ids = suffix_ids[:-1]
        prompt_tail = suffix_ids[-1:]
        context = prefix_ids + document_ids
        answer_ids = self.encode_answer(selected_answer)

        layout = CacheLayout(
            prefix_length=len(prefix_ids),
            data_start=len(prefix_ids),
            block_size=self.block_size,
            n_blocks=len(sample.blocks),
            suffix_length=0,
            original_length=len(context),
            block_content_lengths=tuple(block_content_lengths),
        )
        if layout.data_end != layout.original_length:
            raise AssertionError("Document cache layout must not contain question tokens")
        return PackedSample(
            sample=sample,
            context_ids=torch.tensor([context], dtype=torch.long),
            query_ids=torch.tensor([query_ids], dtype=torch.long),
            prompt_tail_ids=torch.tensor([prompt_tail], dtype=torch.long),
            answer_ids=torch.tensor([answer_ids], dtype=torch.long),
            answer_text=selected_answer,
            answer_aliases=list(selected_aliases),
            layout=layout,
        )
