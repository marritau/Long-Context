from __future__ import annotations

from pathlib import Path
import zipfile


EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", ".venv-check", "dist", "runs"}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    destination = root / "dist" / "boundary_interaction_kv_v18_longbench_2wiki.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    include_roots = ("src", "scripts", "configs", "tests", "notebooks", "docs")
    explicit = ("pyproject.toml", "README.md", "COLAB_KAGGLE_RUNBOOK.md")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in explicit:
            path = root / relative
            if path.exists():
                archive.write(path, Path(root.name) / relative)
        for relative in include_roots:
            base = root / relative
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                    continue
                if any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts):
                    continue
                archive.write(path, Path(root.name) / path.relative_to(root))
    print(destination)


if __name__ == "__main__":
    main()
