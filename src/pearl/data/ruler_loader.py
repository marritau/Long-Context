"""RULER loader placeholder.

Use `JsonlDatasetLoader` after converting official RULER outputs to the PEARL
schema. The class exists so experiment configs can refer to a stable loader.
"""

from pearl.data.base import JsonlDatasetLoader


class RulerLoader(JsonlDatasetLoader):
    dataset_name = "ruler"
