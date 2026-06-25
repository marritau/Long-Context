"""HotpotQA loader placeholder for normalized JSONL files."""

from pearl.data.base import JsonlDatasetLoader


class HotpotLoader(JsonlDatasetLoader):
    dataset_name = "hotpotqa"
