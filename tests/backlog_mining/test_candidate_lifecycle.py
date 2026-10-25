"""AC-SM-1 lifecycle regression coverage."""

from __future__ import annotations

import pytest

from rumen.backlog_mining.models import BacklogCandidate, BacklogCandidateDisposition
from rumen.backlog_mining.repository import (
    CallerPostimageError,
    CandidateRepository,
    MissingRationaleError,
    StaleHeadError,
    UnsupportedActionError,
)


def _candidate(repo: CandidateRepository, fingerprint: str = "family") -> BacklogCandidate:
    return repo.create_candidate(
        fingerprint=fingerprint,
        source_snapshot={"source": "one"},
        content_snapshot={"title": "original"},
        assessment={"assessment": "valid"},
        configuration_snapshot={"target": "rumen"},
    )


def _replacement(
    repo: CandidateRepository, current: BacklogCandidate, title: str
) -> BacklogCandidate:
    return repo.create_candidate(
        fingerprint=current.fingerprint,
        source_snapshot=current.source_snapshot,
        content_snapshot={"title": title},
        assessment=current.assessment,
        configuration_snapshot=current.configuration_snapshot,
        active=False,
    )


def _states(session) -> dict[int, str]:
    return {
        candidate.id: candidate.state
        for candidate in session.query(BacklogCandidate).order_by(BacklogCandidate.id)
    }


def _assert_one_active(session) -> None:
    states = _states(session)
    assert (
        sum(state in {"draft", "accepted", "edited", "deferred"} for state in states.values()) == 1
    )


def test_transition_when_invalid_input_then_returns_typed_error_without_history_or_state_change(
    session,
) -> None:
    repo = CandidateRepository(session)
    candidate = _candidate(repo)
    before = _states(session)
    with pytest.raises(UnsupportedActionError):
        repo.transition(candidate.id, action="close", actor="reviewer", rationale="no")
    with pytest.raises(MissingRationaleError):
        repo.transition(candidate.id, action="accept", actor="reviewer", rationale="")
    with pytest.raises(CallerPostimageError):
        repo.transition(
            candidate.id,
            action="accept",
            actor="reviewer",
            rationale="valid",
            postimage={},
        )
    with pytest.raises(StaleHeadError):
        repo.transition(
            candidate.id,
            action="accept",
            actor="reviewer",
            rationale="valid",
            expected_head="stale",
        )
    assert _states(session) == before
    assert session.query(BacklogCandidateDisposition).count() == 0


def test_lifecycle_when_a_b_c_undo_redo_then_restores_complete_snapshots_and_one_holder(
    session,
) -> None:
    repo = CandidateRepository(session)
    a = _candidate(repo)
    repo.transition(a.id, action="accept", actor="reviewer", rationale="A")
    b = _replacement(repo, a, "B")
    repo.transition(
        a.id,
        action="supersede",
        actor="reviewer",
        rationale="B replaces A",
        replacement_candidate_id=b.id,
        expected_head=repo._applied_head(a.fingerprint).frame_id,
    )
    c = _replacement(repo, a, "C")
    repo.transition(
        b.id,
        action="supersede",
        actor="reviewer",
        rationale="C replaces B",
        replacement_candidate_id=c.id,
        expected_head=repo._applied_head(a.fingerprint).frame_id,
    )
    assert _states(session) == {a.id: "superseded", b.id: "superseded", c.id: "draft"}
    _assert_one_active(session)
    repo.undo(a.fingerprint, actor="reviewer", rationale="undo C")
    assert _states(session) == {a.id: "superseded", b.id: "draft", c.id: "superseded"}
    _assert_one_active(session)
    repo.undo(a.fingerprint, actor="reviewer", rationale="undo B")
    assert _states(session) == {a.id: "accepted", b.id: "superseded", c.id: "superseded"}
    _assert_one_active(session)
    repo.redo(a.fingerprint, actor="reviewer", rationale="redo B")
    assert _states(session) == {a.id: "superseded", b.id: "draft", c.id: "superseded"}
    repo.redo(a.fingerprint, actor="reviewer", rationale="redo C")
    assert _states(session) == {a.id: "superseded", b.id: "superseded", c.id: "draft"}
    _assert_one_active(session)
    assert session.query(BacklogCandidateDisposition).filter_by(record_kind="control").count() == 4


def test_undo_when_later_creation_is_active_then_reconciles_collision_and_redo_restores_later_holder(
    session,
) -> None:
    repo = CandidateRepository(session)
    a = _candidate(repo)
    rejected = repo.transition(a.id, action="reject", actor="reviewer", rationale="not now")
    b = _candidate(repo)
    assert b.id != a.id and b.state == "draft"
    repo.undo(a.fingerprint, actor="reviewer", rationale="restore A")
    assert _states(session) == {a.id: "draft", b.id: "superseded"}
    repo.redo(a.fingerprint, actor="reviewer", rationale="restore rejection")
    assert _states(session) == {a.id: "rejected", b.id: "draft"}
    assert rejected.stack_state == "applied"


@pytest.mark.parametrize("action", ["accept", "edit", "defer", "reopen"])
def test_transition_when_inactive_action_activates_target_then_supersedes_other_holder_and_undoes(
    session, action: str
) -> None:
    repo = CandidateRepository(session)
    a = _candidate(repo)
    repo.transition(a.id, action="reject", actor="reviewer", rationale="inactive")
    b = _candidate(repo)
    frame = repo.transition(a.id, action=action, actor="reviewer", rationale="reactivate")
    assert (
        a.state
        == {"accept": "accepted", "edit": "edited", "defer": "deferred", "reopen": "draft"}[action]
    )
    assert b.state == "superseded"
    assert set(frame.preimage) == {str(a.id), str(b.id)}
    assert set(frame.postimage) == {str(a.id), str(b.id)}
    repo.undo(a.fingerprint, actor="reviewer", rationale="undo")
    assert a.state == "rejected"
    assert b.state == "draft"


@pytest.mark.parametrize("action", ["reject", "knowledge_only", "merge"])
def test_transition_when_nonactive_action_then_accepts_every_action_from_any_state(
    session, action: str
) -> None:
    repo = CandidateRepository(session)
    candidate = _candidate(repo)
    repo.transition(candidate.id, action="accept", actor="reviewer", rationale="accepted")
    repo.transition(candidate.id, action=action, actor="reviewer", rationale="any state")
    assert (
        candidate.state
        == {"reject": "rejected", "knowledge_only": "knowledge_only", "merge": "merged"}[action]
    )


def test_transition_when_new_action_after_undo_then_abandons_redo_future_permanently(
    session,
) -> None:
    repo = CandidateRepository(session)
    candidate = _candidate(repo)
    first = repo.transition(candidate.id, action="accept", actor="reviewer", rationale="first")
    second = repo.transition(candidate.id, action="edit", actor="reviewer", rationale="second")
    repo.undo(candidate.fingerprint, actor="reviewer", rationale="undo second")
    branch = repo.transition(candidate.id, action="defer", actor="reviewer", rationale="branch")
    assert second.stack_state == "inert"
    repo.undo(candidate.fingerprint, actor="reviewer", rationale="undo branch")
    repo.redo(candidate.fingerprint, actor="reviewer", rationale="redo branch")
    assert first.stack_state == "applied"
    assert second.stack_state == "inert"
    assert branch.stack_state == "applied"


def test_classify_revision_when_evidence_cosmetic_shape_or_abstain_then_preserves_required_history(
    session,
) -> None:
    repo = CandidateRepository(session)
    candidate = _candidate(repo)
    repo.append_evidence(candidate.id, [{"kind": "source", "excerpt": "more evidence"}])
    assert session.query(BacklogCandidateDisposition).count() == 0
    cosmetic = repo.classify_revision(
        candidate.id,
        classification="cosmetic",
        proposed_content_snapshot={"title": "wording"},
        actor="reviewer",
        rationale="same scope",
    )
    assert isinstance(cosmetic, BacklogCandidateDisposition)
    assert candidate.content_snapshot == {"title": "wording"}
    assert cosmetic.action_payload["old_content"] == {"title": "original"}
    assert cosmetic.action_payload["new_content"] == {"title": "wording"}
    shaped = repo.classify_revision(
        candidate.id,
        classification="shape_change",
        proposed_content_snapshot={"title": "new scope"},
        actor="reviewer",
        rationale="different outcome",
    )
    assert isinstance(shaped, BacklogCandidateDisposition)
    assert session.query(BacklogCandidate).count() == 2
    assert candidate.state == "superseded"
    family = session.query(BacklogCandidate).order_by(BacklogCandidate.id).all()
    assert family[-1].state == "draft"
    frame_count = session.query(BacklogCandidateDisposition).count()
    repo.classify_revision(
        family[-1].id,
        classification="abstain",
        proposed_content_snapshot={"title": "uncertain"},
        actor="reviewer",
        rationale="uncertain",
    )
    assert session.query(BacklogCandidateDisposition).count() == frame_count
