"""T-05: false-negative audit — drops are logged (never hard-deleted) + surfaced."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from sqlalchemy import select

from rumen.classifier.cascade import classify_pending
from rumen.dropped_items import log_drop, surface_low_confidence_drops
from rumen.grouping import group_pending
from rumen.models import DroppedItem, FeedItem, SkosConcept, TopicEntity
from rumen.taxonomy.corrections import record_classification_review

_TRAIN = [
    ("released cli version install package npm binary", "tool"),
    ("cli release changelog upgrade install binary package", "tool"),
    ("gambling casino betting odds poker slots wager", "gambling"),
    ("casino poker betting gambling odds wager slots", "gambling"),
]


def test_log_drop_persists_row_with_reason_and_confidence(session) -> None:
    row = log_drop(
        session,
        source="hf_daily",
        source_id="p1",
        url="https://arxiv.org/abs/1",
        title="paper",
        reason="classified_out_of_scope",
        drop_confidence=0.4,
    )
    session.commit()
    stored = session.get(DroppedItem, row.id)
    assert stored.drop_reason == "classified_out_of_scope"
    assert stored.drop_confidence == 0.4
    assert stored.reviewed in (0, False)


def test_surface_only_low_confidence_unreviewed_drops(session) -> None:
    log_drop(session, source="s", source_id="lo", reason="low_relevance", drop_confidence=0.2)
    log_drop(
        session, source="s", source_id="hi", reason="classified_out_of_scope", drop_confidence=0.95
    )
    session.commit()

    surfaced = surface_low_confidence_drops(session, max_confidence=0.5)
    ids = {d.source_id for d in surfaced}
    assert "lo" in ids  # low-confidence drop surfaced
    assert "hi" not in ids  # high-confidence drop excluded


def test_out_of_scope_classification_is_logged_not_hard_deleted(session) -> None:
    now = datetime(2026, 7, 10, tzinfo=UTC)
    session.add(
        FeedItem(
            source="reddit",
            source_id="g1",
            url="u",
            title="casino poker betting gambling odds wager slots",
            content="gambling",
            published_at=now,
        )
    )
    gambling = SkosConcept(facet="topic", pref_label="gambling", scheme="test")
    session.add(gambling)
    session.commit()
    llm = type(
        "Llm",
        (),
        {"run": lambda self, prompt: f'{{"concept_ids": [{gambling.id}], "confidence": 0.9}}'},
    )()
    counts = classify_pending(session, llm_command=llm, out_of_scope=frozenset({"gambling"}))
    assert counts["dropped"] == 1
    # The feed_item is NOT hard-deleted — it still exists, now classified.
    item = session.query(FeedItem).filter_by(source_id="g1").one()
    assert item.category == "gambling"
    assert item.status == "dropped"
    # And a dropped_items audit row was written.
    assert session.query(DroppedItem).filter_by(source_id="g1").count() == 1

    counts = group_pending(session, embed=False)
    assert counts == {"entities_created": 0, "identity_merges": 0}
    assert session.query(TopicEntity).count() == 0


def test_given_human_review_routes_an_out_of_scope_item_then_it_matches_live_cascade_drop_semantics(
    session,
) -> None:
    now = datetime(2026, 7, 10, tzinfo=UTC)
    live_item = FeedItem(
        source="reddit",
        source_id="live-drop",
        url="https://example.test/live-drop",
        title="casino poker betting gambling odds wager slots",
        content="gambling",
        published_at=now,
    )
    reviewed_item = FeedItem(
        source="review",
        source_id="human-drop",
        url="https://example.test/human-drop",
        title="Human reviewed gambling item",
        published_at=now,
    )
    gambling = SkosConcept(facet="topic", pref_label="gambling", scheme="test")
    session.add_all([live_item, reviewed_item, gambling])
    reviewed_item.category = "already-classified"
    session.commit()
    llm = type(
        "Llm",
        (),
        {"run": lambda self, prompt: f'{{"concept_ids": [{gambling.id}], "confidence": 0.9}}'},
    )()

    classify_pending(session, llm_command=llm, out_of_scope=frozenset({"gambling"}))
    record_classification_review(
        session,
        item_table="feed_items",
        item_id=reviewed_item.id,
        facet="topic",
        corrected_concept_ids=[gambling.id],
        actor="human:reviewer",
        rationale="This item is out of scope.",
        source="test",
        out_of_scope=frozenset({"gambling"}),
        route="route",
    )
    session.commit()

    live_drop = session.scalar(select(DroppedItem).where(DroppedItem.source_id == "live-drop"))
    human_drop = session.scalar(select(DroppedItem).where(DroppedItem.source_id == "human-drop"))
    assert live_drop is not None and human_drop is not None
    assert (
        human_drop.drop_reason,
        human_drop.drop_confidence,
        human_drop.reviewed,
        human_drop.rescued,
    ) == (
        live_drop.drop_reason,
        1.0,
        live_drop.reviewed,
        live_drop.rescued,
    )
    assert (human_drop.source, human_drop.source_id, human_drop.url, human_drop.title) == (
        reviewed_item.source,
        reviewed_item.source_id,
        reviewed_item.url,
        reviewed_item.title,
    )
    assert session.get(FeedItem, reviewed_item.id).status == "dropped"
    assert session.get(FeedItem, reviewed_item.id) is not None


def test_log_drop_accepts_feeditem_like_namespace(session) -> None:
    # log_drop works from any (source, source_id, url, title) tuple, incl. a rescue re-create.
    fake = SimpleNamespace(source="hn", source_id="h1", url="u", title="t")
    log_drop(
        session,
        source=fake.source,
        source_id=fake.source_id,
        url=fake.url,
        title=fake.title,
        reason="keyword_miss",
        drop_confidence=0.1,
    )
    session.commit()
    assert session.query(DroppedItem).filter_by(source_id="h1").count() == 1
