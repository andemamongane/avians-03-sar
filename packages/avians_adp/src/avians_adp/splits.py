"""Train / val / test by seed.

A seed fixes the scenario (where the person is, the help delay), so every
episode of one seed (any pilot, any tier) goes to the same split: never one
scenario on both sides. The split comes from a hash of the seed, not from
the set of episodes, so adding episodes never moves an old seed and every
build agrees.
"""
from __future__ import annotations

import hashlib
import math

SPLITS = ("train", "val", "test")


def parse_ratios(text: str) -> tuple[float, float, float]:
    """'80/10/10' or '0.8,0.1,0.1' -> fractions summing to 1."""
    parts = [float(x) for x in text.replace(",", "/").split("/")]
    if len(parts) != 3 or not all(math.isfinite(p) and p >= 0 for p in parts) or sum(parts) <= 0:
        raise ValueError(f"split ratios must be three non-negative numbers, got {text!r}")
    total = sum(parts)
    return tuple(p / total for p in parts)  # type: ignore[return-value]


def check_seed(seed) -> int:
    """Seeds are integers. Refuse anything else rather than coerce it: 1.9,
    True and "07" would otherwise silently share seed 1's or 7's split."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    return seed


def unit(seed: int) -> float:
    """A stable value in [0, 1) for a seed: the same on every machine and Python."""
    h = hashlib.sha256(f"avians-split:{check_seed(seed)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def split_of(seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    u = unit(seed)
    edge = 0.0
    for name, r in zip(SPLITS, ratios):
        edge += r
        if u < edge:
            return name
    return next(n for n, r in zip(reversed(SPLITS), reversed(ratios)) if r > 0)   # u == 1 - tiny rounding
