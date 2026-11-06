"""What McGlashan's package must do next: these turn on as the pieces land."""
import pytest

from avians_adp import sft

pytestmark = pytest.mark.skip(reason="McGlashan: implement avians_adp.sft (README.md, 'Your tasks')")


def test_pilot_examples_one_per_decision():
    assert sft.pilot_examples


def test_router_examples():
    assert sft.router_examples
