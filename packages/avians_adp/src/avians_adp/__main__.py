"""python -m avians_adp validate <episodes dir> [...]    check every trajectory.json under the folders"""
from __future__ import annotations

import sys

from .validation import validate_dir


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or args[0] != "validate":
        print(__doc__)
        return 2
    bad = 0
    for root in args[1:]:
        for ep, problems in validate_dir(root).items():
            if problems:
                bad += 1
                print(f"{ep}:")
                for p in problems:
                    print(f"  - {p}")
            else:
                print(f"{ep}: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
