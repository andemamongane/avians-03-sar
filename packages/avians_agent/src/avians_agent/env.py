"""API keys from .env at the repository root."""
from __future__ import annotations

import os

from .mission import repo_root


def load_env() -> None:
    """Read KEY=VALUE lines from .env at the repo root into the environment (not overriding)."""
    env = repo_root() / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, eq, v = line.strip().partition("=")
            if eq and k and not k.startswith("#") and v and k not in os.environ:
                os.environ[k] = v
