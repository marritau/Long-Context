"""LIBRA loader placeholder for normalized JSONL files."""

from pearl.data.base import JsonlDatasetLoader


class LibraLoader(JsonlDatasetLoader):
    dataset_name = "libra"
