"""What McGlashan's package must do next: these turn on as the pieces land.

test_pilot_examples_one_per_decision is now a real test in test_sft.py (Phase 2).
"""
import pytest
from avians_adp import sft

pytestmark = pytest.mark.skip(reason="McGlashan: implement avians_adp.sft.router_examples (PLAN.md, Phase 4)")


def test_router_examples():
    assert sft.router_examples
