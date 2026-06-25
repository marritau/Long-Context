"""NoLiMa loader placeholder for normalized JSONL files."""

from pearl.data.base import JsonlDatasetLoader


class NoLiMaLoader(JsonlDatasetLoader):
    dataset_name = "nolima"
