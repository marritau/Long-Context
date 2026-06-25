"""Answer quality metrics."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(answer: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace."""

    answer = answer.lower()
    answer = "".join(ch if ch not in string.punctuation else " " for ch in answer)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def exact_match(prediction: str, gold: str | list[str]) -> float:
    """Exact match against one or more gold answers."""

    golds = gold if isinstance(gold, list) else [gold]
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(item) for item in golds))


def token_f1(prediction: str, gold: str | list[str]) -> float:
    """Maximum token F1 against one or more gold answers."""

    golds = gold if isinstance(gold, list) else [gold]
    return max((_token_f1_one(prediction, item) for item in golds), default=0.0)


def _token_f1_one(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
