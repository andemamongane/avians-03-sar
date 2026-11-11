from pathlib import Path

from avians_skills.docs import render


def test_action_space_doc_matches_the_registry():
    repo = Path(__file__).resolve().parents[3]
    assert (repo / "docs/action_space.md").read_text() == render(), \
        "docs/action_space.md is stale: python -m avians_skills.docs > docs/action_space.md"
