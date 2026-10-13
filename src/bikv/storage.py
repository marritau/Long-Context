from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any


class ResultStore:
    def __init__(self, root: str | Path, run_id: str, mirror_root: str | Path | None = None):
        self.root = Path(root).resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.mirror = Path(mirror_root).resolve() / run_id if mirror_root else None
        if self.mirror:
            self.mirror.mkdir(parents=True, exist_ok=True)

    def path(self, namespace: str, key: str, suffix: str = ".json") -> Path:
        return self.root / namespace / f"{key}{suffix}"

    def exists(self, namespace: str, key: str, suffix: str = ".json") -> bool:
        path = self.path(namespace, key, suffix)
        if path.exists():
            return True
        if self.mirror:
            mirrored = self.mirror / namespace / f"{key}{suffix}"
            if mirrored.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mirrored, path)
                return True
        return False

    def read_json(self, namespace: str, key: str, suffix: str = ".json") -> dict[str, Any]:
        if not self.exists(namespace, key, suffix):
            raise FileNotFoundError(self.path(namespace, key, suffix))
        path = self.path(namespace, key, suffix)
        opener = gzip.open if suffix.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(self, namespace: str, key: str, value: dict[str, Any], compress: bool = False) -> Path:
        suffix = ".json.gz" if compress else ".json"
        destination = self.path(namespace, key, suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        opener = gzip.open if compress else open
        with opener(temporary, "wt", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            if not compress:
                os.fsync(handle.fileno())
        temporary.replace(destination)
        self._mirror_file(destination)
        return destination

    def _mirror_file(self, source: Path) -> None:
        if not self.mirror:
            return
        relative = source.relative_to(self.root)
        destination = self.mirror / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)

    def write_status(self, **values: Any) -> None:
        self.write_json("", "status", values)

    def write_text(self, namespace: str, key: str, value: str, suffix: str = ".md") -> Path:
        destination = self.path(namespace, key, suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        self._mirror_file(destination)
        return destination

    def bundle(self, destination: str | Path | None = None) -> Path:
        output = Path(destination) if destination else self.root.parent / f"{self.root.name}_artifacts.zip"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.root.rglob("*")):
                if path.is_file() and not path.name.endswith(".tmp"):
                    archive.write(path, path.relative_to(self.root.parent))
        temporary.replace(output)
        if self.mirror:
            mirror_bundle = self.mirror.parent / output.name
            mirror_temporary = mirror_bundle.with_name(mirror_bundle.name + ".tmp")
            shutil.copy2(output, mirror_temporary)
            mirror_temporary.replace(mirror_bundle)
        return output


def stable_mask_key(selected: list[int]) -> str:
    canonical = ",".join(str(index) for index in sorted(selected))
    digest = hashlib.sha1(canonical.encode("ascii")).hexdigest()[:12]
    return f"mask-{digest}"
