"""Review-surface regression coverage."""

from __future__ import annotations

import pytest

from rumen.backlog_mining.repository import CandidateRepository
from rumen.backlog_mining.review import (
    CandidateRenderError,
    candidate_review,
    render_candidate,
    review_candidate,
)
from rumen.cli import run
from rumen.models import (
    BacklogCandidateDisposition,
    BacklogCandidateRecommendation,
)


def _candidate(session, *, state: str = "draft", valid_payload: bool = True):
    candidate = CandidateRepository(session).create_candidate(
        fingerprint="review-family",
        source_snapshot={"source_claims": [{"url": "https://example.test/source"}]},
        content_snapshot={"title": "Review this proposal", "claim_ids": ["source:1"]}
        if valid_payload
        else {"claim_ids": ["source:1"]},
        assessment={"claims": [{"id": "source:1", "kind": "source"}]},
        configuration_snapshot={
            "beads_repo_identity": "owner/repo",
            "task_store": "queue",
            "template_revision": "v1",
        },
        evidence=[
            {
                "kind": "source",
                "excerpt": "quoted source claim",
                "metadata": {"url": "https://example.test/source"},
            },
            {"kind": "internal_context", "excerpt": "configured context"},
            {"kind": "inference", "excerpt": "model inference"},
        ],
    )
    candidate.state = state
    session.add(
        BacklogCandidateRecommendation(
            candidate_id=candidate.id,
            task_store="queue",
            recommended_cc_task=1,
            rationale="review in the current session",
            confidence=0.9,
        )
    )
    session.flush()
    return candidate


def test_candidate_review_when_shown_then_separates_evidence_context_links_history_and_recommendations(
    session,
) -> None:
    candidate = _candidate(session)

    review = candidate_review(session, str(candidate.id))

    assert review["candidate"]["fingerprint"] == candidate.fingerprint
    assert review["source_claims"]["source"][0]["excerpt"] == "quoted source claim"
    assert review["internal_context"]["configuration"]["task_store"] == "queue"
    assert review["model_inferences"]["inference"][0]["excerpt"] == "model inference"
    assert review["task_matches"] == []
    assert review["disposition_history"] == []
    assert review["recommendations"][0]["recommended_cc_task"] is True
    assert review["recommendations"][0]["recommended_repo"] == "owner/repo"


def test_review_candidate_when_valid_disposition_then_records_actor_and_rationale(session) -> None:
    candidate = _candidate(session)

    frame = review_candidate(
        session,
        str(candidate.id),
        action="accept",
        actor="reviewer@example.test",
        rationale="evidence supports a scoped draft",
    )

    assert candidate.state == "accepted"
    assert frame.actor == "reviewer@example.test"
    assert frame.rationale == "evidence supports a scoped draft"


def test_review_candidate_when_invalid_action_then_leaves_candidate_and_history_unchanged(
    session,
) -> None:
    candidate = _candidate(session)

    with pytest.raises(ValueError, match="unsupported candidate action"):
        review_candidate(
            session,
            str(candidate.id),
            action="invalid",
            actor="reviewer",
            rationale="no",
        )

    assert candidate.state == "draft"
    assert session.query(BacklogCandidateDisposition).count() == 0


def test_backlog_cli_when_show_or_invalid_review_then_renders_sections_or_returns_clean_error(
    capsys, session
) -> None:
    candidate = _candidate(session)

    assert run(["backlog", "show", candidate.fingerprint], session) == 0
    shown = capsys.readouterr().out
    assert '"source_claims"' in shown
    assert '"disposition_history"' in shown

    assert (
        run(
            [
                "backlog",
                "review",
                str(candidate.id),
                "--action",
                "invalid",
                "--actor",
                "reviewer",
                "--rationale",
                "no",
            ],
            session,
        )
        == 1
    )
    assert "error: unsupported candidate action" in capsys.readouterr().err
    assert candidate.state == "draft"


@pytest.mark.parametrize(
    ("state", "valid_payload", "message"),
    [
        ("draft", True, "accepted or edited"),
        ("accepted", False, "non-empty title"),
    ],
)
def test_render_candidate_when_not_approved_or_payload_is_invalid_then_raises_without_mutation(
    session, state: str, valid_payload: bool, message: str
) -> None:
    candidate = _candidate(session, state=state, valid_payload=valid_payload)
    before_history = session.query(BacklogCandidateDisposition).count()

    with pytest.raises(CandidateRenderError, match=message):
        render_candidate(session, candidate.id)

    assert candidate.state == state
    assert session.query(BacklogCandidateDisposition).count() == before_history


def test_render_candidate_when_approved_then_returns_title_body_and_evidence_links(session) -> None:
    candidate = _candidate(session, state="accepted")

    draft = render_candidate(session, candidate.id)

    assert draft.title == "Review this proposal"
    assert "Review this proposal" in draft.body
    assert draft.evidence_links == ("https://example.test/source",)
