from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    values: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        config_path = Path(path).resolve()
        values = json.loads(config_path.read_text(encoding="utf-8"))
        required = {
            "run_id",
            "seed",
            "model_name",
            "n_stories",
            "n_blocks",
            "block_size",
            "budget_blocks",
            "data_path",
            "output_root",
        }
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        if not 1 <= int(values["budget_blocks"]) <= int(values["n_blocks"]):
            raise ValueError("budget_blocks must be between 1 and n_blocks")
        return cls(config_path, values)

    @property
    def root(self) -> Path:
        return self.path.parent.parent

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def resolve(self, key: str) -> Path:
        value = Path(str(self.values[key]))
        return value if value.is_absolute() else (self.root / value).resolve()

    @property
    def config_hash(self) -> str:
        ignored = {
            "output_root",
            "mirror_root",
            "data_path",
            "max_gpu_minutes",
            "shutdown_reserve_minutes",
            "bundle_every_samples",
            "bundle_after_each_mask",
        }
        scientific = {key: value for key, value in self.values.items() if key not in ignored}
        payload = json.dumps(scientific, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_manifest(self) -> dict[str, Any]:
        return {"config_path": str(self.path), "config_hash": self.config_hash, "config": self.values}
