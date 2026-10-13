from __future__ import annotations

import zipfile
from pathlib import Path


EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    ".venv-check",
    "__pycache__",
    "dist",
    "paper",
    "runs",
    "tmp",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    destination = root / "dist" / "boundary_interaction_kv_v16_continuation_suite.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file():
                continue
            if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            packaged_data = relative.parts[:2] in {
                ("data", "v15_g1b_pilot"),
                ("data", "v15_g1b_confirmation"),
            }
            if relative.parts[:1] == ("data",) and path.suffix in {".json", ".jsonl"} and not packaged_data:
                continue
            archive.write(path, Path("boundary_interaction_kv") / relative)
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
