"""Position utility sweep helper."""

from __future__ import annotations

import argparse

from pearl.packing.position_utility import lost_middle_utility
from pearl.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/metrics/position_utility.json")
    parser.add_argument("--rho", type=float, default=0.35)
    parser.add_argument("--kappa", type=float, default=2.0)
    args = parser.parse_args()

    positions = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    rows = {
        str(position): lost_middle_utility(position, rho=args.rho, kappa=args.kappa)
        for position in positions
    }
    write_json(args.out, rows)


if __name__ == "__main__":
    main()
