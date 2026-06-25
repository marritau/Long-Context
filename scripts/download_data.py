"""Dataset download scaffold.

The script intentionally does not auto-download large external datasets in the
MVP. Add dataset-specific download/normalization code here once credentials,
storage, and benchmark versions are fixed.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(
        f"Download scaffold for {args.dataset}. "
        "Place raw files under data/raw/ and normalize them to PEARL JSONL."
    )


if __name__ == "__main__":
    main()
