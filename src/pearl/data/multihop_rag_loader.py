"""MultiHop-RAG loader placeholder for normalized JSONL files."""

from pearl.data.base import JsonlDatasetLoader


class MultiHopRagLoader(JsonlDatasetLoader):
    dataset_name = "multihop_rag"
