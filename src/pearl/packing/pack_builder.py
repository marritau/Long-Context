"""End-to-end PEARL context package builder."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pearl.compression.query_aware import query_aware_compress
from pearl.data.schema import LongContextExample, PackedChunk, PackedContext
from pearl.graph.evidence_graph import build_evidence_graph
from pearl.graph.pagerank import personalized_pagerank
from pearl.packing.ordering import position_aware_order
from pearl.packing.prompt_templates import build_qa_prompt
from pearl.packing.selectors import mmr_select
from pearl.preprocessing.chunking import chunk_documents
from pearl.preprocessing.token_counting import count_tokens
from pearl.retrieval.bm25 import BM25Index
from pearl.retrieval.dense import HashingDenseRetriever
from pearl.retrieval.hybrid import combine_scores, hybrid_rrf_ids
from pearl.retrieval.rerank import rerank_scores

DEFAULT_CONFIG: dict[str, Any] = {
    "chunking": {"chunk_size": 768, "chunk_overlap": 128, "structure_aware": True},
    "retrieval": {
        "top_m": 80,
        "top_k": 12,
        "bm25_weight": 0.20,
        "dense_weight": 0.35,
        "cross_encoder_weight": 0.35,
        "structure_weight": 0.10,
        "rrf_k": 60,
        "dense_dim": 384,
    },
    "graph": {
        "enabled": True,
        "semantic_weight": 0.60,
        "adjacency_weight": 0.25,
        "section_weight": 0.15,
        "pagerank_alpha": 0.85,
        "pagerank_iters": 50,
    },
    "selection": {
        "redundancy_penalty": 0.25,
        "relevance_weight": 0.55,
        "graph_weight": 0.45,
        "pre_compression_budget": 24000,
    },
    "compression": {
        "enabled": True,
        "final_budget": 8192,
        "min_chunk_budget": 128,
        "sentence_neighbor_keep": True,
        "compression_temperature": 1.0,
    },
    "ordering": {"enabled": True, "mode": "pearl", "rho": 0.35, "kappa": 2.0},
}


def pearl_pack(
    example: LongContextExample,
    config: dict[str, Any] | None = None,
    method: str = "pearl_full",
) -> tuple[str, PackedContext]:
    """Build a PEARL evidence package and prompt for one example."""

    cfg = _merge_config(DEFAULT_CONFIG, config or {})
    chunks = chunk_documents(
        example.documents,
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["chunk_overlap"],
        structure_aware=cfg["chunking"]["structure_aware"],
    )
    if not chunks:
        packed = PackedContext(
            example_id=example.example_id,
            method=method,
            chunks=[],
            total_tokens=0,
            compression_ratio=0.0,
            ordering_strategy=cfg["ordering"]["mode"],
        )
        return build_qa_prompt(example.question, []), packed

    bm25 = BM25Index(chunks)
    bm25_scores = bm25.score(example.question)
    dense = HashingDenseRetriever(chunks, dim=cfg["retrieval"].get("dense_dim", 384))
    dense_scores = dense.score(example.question)

    candidate_ids = hybrid_rrf_ids(
        bm25_scores,
        dense_scores,
        top_m=min(cfg["retrieval"]["top_m"], len(chunks)),
        rrf_k=cfg["retrieval"]["rrf_k"],
    )
    candidate_chunks = [chunks[idx] for idx in candidate_ids]
    candidate_ce_scores = rerank_scores(example.question, candidate_chunks)
    ce_scores = [0.0] * len(chunks)
    for idx, score in zip(candidate_ids, candidate_ce_scores, strict=True):
        ce_scores[idx] = score

    structure_scores = _compute_structure_scores(example.question, chunks, bm25_scores)
    relevance = combine_scores(
        {
            "bm25": bm25_scores,
            "dense": dense_scores,
            "ce": ce_scores,
            "structure": structure_scores,
        },
        {
            "bm25": cfg["retrieval"]["bm25_weight"],
            "dense": cfg["retrieval"]["dense_weight"],
            "ce": cfg["retrieval"]["cross_encoder_weight"],
            "structure": cfg["retrieval"]["structure_weight"],
        },
    )

    if cfg["graph"].get("enabled", True):
        graph = build_evidence_graph(
            chunks,
            vectors=dense.chunk_vectors,
            semantic_weight=cfg["graph"]["semantic_weight"],
            adjacency_weight=cfg["graph"]["adjacency_weight"],
            section_weight=cfg["graph"]["section_weight"],
        )
        graph_scores = personalized_pagerank(
            graph,
            seed=relevance,
            alpha=cfg["graph"]["pagerank_alpha"],
            iters=cfg["graph"]["pagerank_iters"],
        )
    else:
        graph_scores = [0.0] * len(chunks)

    selected = mmr_select(
        chunks=chunks,
        relevance=relevance,
        graph_scores=graph_scores,
        budget=cfg["selection"]["pre_compression_budget"],
        vectors=dense.chunk_vectors,
        redundancy_penalty=cfg["selection"]["redundancy_penalty"],
        relevance_weight=cfg["selection"]["relevance_weight"],
        graph_weight=cfg["selection"]["graph_weight"],
        max_chunks=cfg["retrieval"].get("top_k"),
    )
    index_by_id = {chunk.chunk_id: idx for idx, chunk in enumerate(chunks)}
    selected_importance = [
        relevance[index_by_id[chunk.chunk_id]] + graph_scores[index_by_id[chunk.chunk_id]]
        for chunk in selected
    ]

    if cfg["compression"].get("enabled", True):
        packed_chunks = query_aware_compress(
            question=example.question,
            chunks=selected,
            total_budget=cfg["compression"]["final_budget"],
            importances=selected_importance,
            min_chunk_budget=cfg["compression"]["min_chunk_budget"],
            keep_neighbors=cfg["compression"]["sentence_neighbor_keep"],
            temperature=cfg["compression"]["compression_temperature"],
        )
    else:
        packed_chunks = [
            PackedChunk(
                chunk=chunk,
                text=chunk.text,
                token_len=chunk.token_len,
                original_token_len=chunk.token_len,
                score=importance,
            )
            for chunk, importance in zip(selected, selected_importance, strict=True)
        ]

    if cfg["ordering"].get("enabled", True):
        ordered = position_aware_order(
            packed_chunks,
            importances=selected_importance,
            mode=cfg["ordering"]["mode"],
            rho=cfg["ordering"]["rho"],
            kappa=cfg["ordering"]["kappa"],
        )
    else:
        ordered = position_aware_order(packed_chunks, mode="document")

    total_tokens = sum(chunk.token_len for chunk in ordered)
    original_tokens = sum(chunk.original_token_len for chunk in ordered)
    packed_context = PackedContext(
        example_id=example.example_id,
        method=method,
        chunks=ordered,
        total_tokens=total_tokens,
        compression_ratio=total_tokens / original_tokens if original_tokens else 0.0,
        ordering_strategy=cfg["ordering"]["mode"] if cfg["ordering"].get("enabled", True) else "document",
        metadata={
            "num_input_chunks": len(chunks),
            "num_selected_chunks": len(selected),
            "prompt_tokens": count_tokens(build_qa_prompt(example.question, ordered)),
        },
    )
    return build_qa_prompt(example.question, ordered), packed_context


def _compute_structure_scores(question: str, chunks, bm25_scores: list[float]) -> list[float]:
    title_chunks = [deepcopy(chunk) for chunk in chunks]
    for chunk in title_chunks:
        chunk.text = " ".join(part for part in [chunk.section_title, chunk.metadata.get("source_title")] if part)
    title_index = BM25Index(title_chunks)
    title_scores = title_index.score(question)
    scores: list[float] = []
    for idx, chunk in enumerate(chunks):
        neighbor = 0.0
        if idx > 0 and chunks[idx - 1].doc_id == chunk.doc_id:
            neighbor = max(neighbor, bm25_scores[idx - 1])
        if idx + 1 < len(chunks) and chunks[idx + 1].doc_id == chunk.doc_id:
            neighbor = max(neighbor, bm25_scores[idx + 1])
        section_bonus = 0.1 if (chunk.section_title or "").lower() in {"abstract", "intro", "conclusion", "method", "results"} else 0.0
        scores.append(title_scores[idx] + 0.25 * neighbor + section_bonus)
    return scores


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged
