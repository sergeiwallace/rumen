"""Stored-Markdown backlog review links."""

from __future__ import annotations

from rumen.backlog_mining.review import append_candidate_review_references


def test_digest_when_candidates_are_referenced_then_includes_identifier_and_read_only_show_command() -> (
    None
):
    markdown = append_candidate_review_references(
        "# Digest\n",
        [{"id": 12, "fingerprint": "fingerprint-12"}],
    )

    assert "fingerprint-12" in markdown
    assert "rumen backlog show fingerprint-12" in markdown
    assert "review" not in markdown.lower()
    assert "http" not in markdown.lower()
