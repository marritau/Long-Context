"""Qasper loader placeholder for normalized JSONL files."""

from pearl.data.base import JsonlDatasetLoader


class QasperLoader(JsonlDatasetLoader):
    dataset_name = "qasper"
