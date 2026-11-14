"""Retention removes copied content through real database foreign keys."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Column, ForeignKey, Integer, Table, select

from rumen.adapters.reddit_subreddits import RedditSubredditSettings
from rumen.models import (
    BacklogCandidate,
    BacklogCandidateDisposition,
    BacklogCandidateEvidence,
    BacklogCandidatePromotion,
    BacklogCandidateRecommendation,
    BacklogCandidateTaskLink,
    BacklogMappingReview,
    BacklogMiningRun,
    BacklogTaskMatch,
    Base,
    ClassificationCorrection,
    Digest,
    DigestEntity,
    DroppedItem,
    FeedFeedback,
    FeedItem,
    ItemConcept,
    MergeCandidate,
    MergeLog,
    SkosConcept,
    TopicEntity,
    YouTubeVideoState,
)
from rumen.reddit_retention import RedditRetentionMaintainer

_CANARY = "retention-content-canary"


class Gateway:
    def revalidate(self, fullnames):
        return iter(())


def _maintainer(gateway):
    return RedditRetentionMaintainer(
        RedditSubredditSettings(
            "example", ("Test",), ("submissions",), "ID", "SECRET", "ua", "approval"
        ),
        gateway,
    )


def _item(source_id, published_at):
    return FeedItem(
        source="reddit:example",
        source_id=source_id,
        title=_CANARY,
        content=_CANARY,
        url="https://www.reddit.com/r/Test/comments/a",
        published_at=published_at,
    )


def _candidate(fingerprint, **references):
    return BacklogCandidate(
        fingerprint=fingerprint,
        source_snapshot={"title": _CANARY},
        content_snapshot={"body": _CANARY},
        assessment={},
        configuration_snapshot={},
        state="accepted",
        **references,
    )


def _correction(table, item_id, **values):
    return ClassificationCorrection(
        item_table=table,
        item_id=item_id,
        facet="topic",
        action="correct",
        prior_concept_ids=[],
        corrected_concept_ids=[],
        actor="human:reviewer",
        source="test",
        rationale=_CANARY,
        **values,
    )


@pytest.fixture
def retention_graph(session):
    now = datetime.now(UTC)
    target = TopicEntity(
        canonical_title=_CANARY,
        display_title=_CANARY,
        canonical_url="https://example.com/target",
        earliest_seen=now,
    )
    other = TopicEntity(
        canonical_title="unrelated", canonical_url="https://example.com/other", earliest_seen=now
    )
    session.add_all([target, other])
    session.flush()
    item = _item("t3_target", now - timedelta(hours=49))
    item.title = item.content = _CANARY
    item.topic_entity_id = target.id
    saved = FeedItem(
        source="article",
        source_id="saved",
        url="https://example.com/saved",
        title="saved",
        published_at=now,
        topic_entity_id=other.id,
    )
    digest = Digest(
        period_start=now,
        period_end=now,
        content_json={"text": _CANARY},
        content_markdown=_CANARY,
    )
    run = BacklogMiningRun(status="completed")
    concept = SkosConcept(facet="topic", pref_label="Example", scheme="test")
    session.add_all([item, saved, digest, run, concept])
    session.flush()
    entity_candidate = _candidate("entity", topic_entity_id=target.id)
    evidence_candidate = _candidate("evidence")
    digest_candidate = _candidate("digest", digest_id=digest.id)
    sibling = _candidate("entity")
    sibling.state = "superseded"
    clean = BacklogCandidate(
        fingerprint="unrelated",
        source_snapshot={},
        content_snapshot={},
        assessment={},
        configuration_snapshot={},
        topic_entity_id=other.id,
    )
    session.add_all([entity_candidate, evidence_candidate, digest_candidate, sibling, clean])
    session.flush()
    # Self references carry no text. They must be detached without deleting clean candidates.
    clean.superseded_by_candidate_id = entity_candidate.id
    clean.merge_target_candidate_id = entity_candidate.id
    rows = [
        BacklogMappingReview(
            mining_run_id=run.id,
            topic_entity_id=target.id,
            status="reviewed",
            confidence=0.9,
            threshold=0.7,
            decision_inputs={"excerpt": _CANARY},
        ),
        BacklogCandidateEvidence(
            candidate_id=evidence_candidate.id,
            feed_item_id=item.id,
            kind="source",
            excerpt=_CANARY,
        ),
        DigestEntity(digest_id=digest.id, topic_entity_id=target.id),
        DigestEntity(digest_id=digest.id, topic_entity_id=other.id),
        FeedFeedback(topic_entity_id=target.id, rating=1),
        MergeLog(
            topic_entity_id=target.id,
            feed_item_id=item.id,
            match_method="manual",
            actor="human",
            evidence={"excerpt": _CANARY},
        ),
        MergeLog(
            topic_entity_id=target.id,
            feed_item_id=saved.id,
            match_method="manual",
            actor="human",
            evidence={"excerpt": _CANARY},
        ),
        MergeCandidate(
            entity_a_id=target.id,
            entity_b_id=other.id,
            status="rejected",
            resolved_by="human:reviewer",
            reason=_CANARY,
        ),
        MergeCandidate(
            entity_a_id=other.id,
            entity_b_id=target.id,
            status="accepted",
            resolved_by="human:reviewer",
            reason=_CANARY,
        ),
        DroppedItem(source=item.source, source_id=item.source_id, title=_CANARY, drop_reason="x"),
        YouTubeVideoState(feed_item_id=item.id, playlist_id="test", video_id="target"),
    ]
    transcript_state = YouTubeVideoState(
        feed_item_id=saved.id,
        transcript_feed_item_id=item.id,
        playlist_id="test",
        video_id="saved",
        transcript_status="available",
        transcript_attempts=2,
    )
    rows.append(transcript_state)
    for candidate in (entity_candidate, evidence_candidate, digest_candidate, sibling):
        rows.extend(
            [
                BacklogCandidateEvidence(
                    candidate_id=candidate.id, kind="snapshot", excerpt=_CANARY
                ),
                BacklogTaskMatch(
                    candidate_id=candidate.id,
                    task_store="example",
                    external_task_key="task",
                    title=_CANARY,
                    state="open",
                ),
                BacklogCandidateRecommendation(
                    candidate_id=candidate.id, task_store="example", rationale=_CANARY
                ),
                BacklogCandidateDisposition(
                    frame_id=candidate.fingerprint + str(candidate.id),
                    fingerprint=candidate.fingerprint,
                    sequence_no=1,
                    action="accept",
                    target_candidate_id=candidate.id,
                    affected_candidate_ids=[candidate.id],
                    preimage={"text": _CANARY},
                    postimage={},
                    actor="human:reviewer",
                    rationale=_CANARY,
                ),
                BacklogCandidatePromotion(
                    operation_id=str(candidate.id),
                    candidate_id=candidate.id,
                    fingerprint=candidate.fingerprint,
                    actor="human:reviewer",
                    requested_payload={"body": _CANARY},
                ),
            ]
        )
    # Undo controls have no candidate FK, but still contain whole-family snapshots.
    rows.append(
        BacklogCandidateDisposition(
            frame_id="undo",
            fingerprint="entity",
            sequence_no=2,
            action="undo",
            preimage={str(entity_candidate.id): {"content_snapshot": {"body": _CANARY}}},
            postimage={},
            actor="human:reviewer",
            rationale="undo",
        )
    )
    # A different family's frame can include this candidate through a JSON snapshot.
    rows.append(
        BacklogCandidateDisposition(
            frame_id="cross-family",
            fingerprint="unrelated",
            sequence_no=1,
            action="merge",
            target_candidate_id=clean.id,
            affected_candidate_ids=[entity_candidate.id],
            preimage={str(entity_candidate.id): {"content_snapshot": {"body": _CANARY}}},
            postimage={},
            actor="human:reviewer",
            rationale="merge",
        )
    )
    for index, (origin, reconciled) in enumerate(
        ((entity_candidate, clean), (clean, entity_candidate))
    ):
        rows.append(
            BacklogCandidateTaskLink(
                fingerprint="link" + str(index),
                task_store="example",
                external_task_key=str(index),
                origin_candidate_id=origin.id,
                last_reconciled_candidate_id=reconciled.id,
                last_observed_title=_CANARY,
            )
        )
    for table, row_id in (("feed_items", item.id), ("topic_entities", target.id)):
        rows.append(
            ItemConcept(
                item_table=table,
                item_id=row_id,
                concept_id=concept.id,
                facet="topic",
                confidence=1.0,
                method="human:reviewer",
            )
        )
        correction = _correction(table, row_id)
        session.add(correction)
        session.flush()
        rows.extend([correction, _correction(table, row_id, supersedes=correction.id)])
    session.add_all(rows)
    session.commit()
    return target.id, other.id, item.id, saved.id, clean.id, transcript_state.id


@pytest.mark.parametrize("entrypoint", ["expired", "missing", "terminate"])
@pytest.mark.parametrize("shared_entity", [False, True])
def test_given_all_content_references_when_reddit_is_purged_then_copies_and_dangling_links_are_removed(
    session, retention_graph, entrypoint, shared_entity
):
    target_id, other_id, item_id, saved_id, clean_id, transcript_state_id = retention_graph
    if shared_entity:
        saved = session.get(FeedItem, saved_id)
        saved.topic_entity_id = target_id
    if entrypoint == "missing":
        session.get(FeedItem, item_id).published_at = datetime.now(UTC)
    session.commit()
    maintainer = _maintainer(Gateway())

    if entrypoint == "terminate":
        assert maintainer.terminate(session) == 1
    else:
        run = maintainer.maintain(session)
        assert run.status == "completed"
        assert run.items_purged == 1
        assert run.reason_counts[entrypoint] == 1

    session.expire_all()
    assert session.get(FeedItem, item_id) is None
    assert session.get(FeedItem, saved_id).title == "saved"
    assert session.get(TopicEntity, other_id).canonical_title == "unrelated"
    target = session.get(TopicEntity, target_id)
    if shared_entity:
        assert target.canonical_title == "saved"
        assert target.display_title is None
        assert session.scalar(select(FeedFeedback.rating)) == 1
    else:
        assert target is None
        assert session.scalar(select(FeedFeedback.id)) is None
    clean = session.get(BacklogCandidate, clean_id)
    assert clean.content_snapshot == {}
    assert clean.superseded_by_candidate_id is None
    assert clean.merge_target_candidate_id is None
    assert list(session.scalars(select(BacklogCandidate.id))) == [clean_id]
    state = session.get(YouTubeVideoState, transcript_state_id)
    assert state.transcript_feed_item_id is None
    assert (state.transcript_status, state.transcript_attempts) == ("available", 2)
    assert list(session.scalars(select(YouTubeVideoState.id))) == [transcript_state_id]
    # Check the whole stored graph, including free text and JSON undo snapshots.
    for table in Base.metadata.tables.values():
        assert _CANARY not in str(session.execute(select(table)).all()), table.name
        for fk in table.foreign_keys:
            dangling = session.execute(
                select(fk.parent).where(fk.parent.is_not(None), ~fk.parent.in_(select(fk.column)))
            ).first()
            assert dangling is None, str(fk)
    assert maintainer.terminate(session) == 0


def test_given_metadata_foreign_keys_when_purge_policies_are_checked_then_every_reference_is_reviewed():
    from rumen.reddit_retention import PURGE_REFERENCE_POLICIES

    references = {
        (table.name, fk.parent.name)
        for table in Base.metadata.tables.values()
        for fk in table.foreign_keys
        if fk.column.table.name
        in {
            "feed_items",
            "topic_entities",
            "backlog_candidates",
            "digests",
            "classification_corrections",
        }
    }
    assert references == set(PURGE_REFERENCE_POLICIES)


@pytest.mark.parametrize(
    "target", ["feed_items", "topic_entities", "backlog_candidates", "digests"]
)
def test_given_unreviewed_foreign_key_when_termination_runs_then_purge_is_refused(session, target):
    item = _item("t3_target", datetime.now(UTC))
    session.add(item)
    session.commit()
    table = Table(
        "unreviewed_retention_reference",
        Base.metadata,
        Column("id", Integer, primary_key=True),
        Column("target_id", ForeignKey(f"{target}.id")),
    )
    try:
        with pytest.raises(ValueError, match="purge policy"):
            _maintainer(Gateway()).terminate(session)
        assert session.get(FeedItem, item.id) is not None
    finally:
        Base.metadata.remove(table)
