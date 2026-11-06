import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
os.environ.setdefault("AVIANS_ROOT", str(ROOT))


@pytest.fixture(scope="session")
def world():
    from avians_env import WorldSpec
    return WorldSpec.load(ROOT / "worlds/sar_valley")


@pytest.fixture(scope="session")
def limits():
    from avians_safety import load
    return load(ROOT / "configs/limits.yaml")[0]
