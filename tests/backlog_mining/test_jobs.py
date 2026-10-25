from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rumen.backlog_mining.task_store import TaskInventory
from rumen.config import PipelineConfig, TargetTemplate
from rumen.jobs import _record_mining_alert, run_backlog_mining
from rumen.models import BacklogMiningRun, FeedItem, TopicEntity


class _ExplodingJudge:
    def evaluate(self, _packet, _target):
        raise RuntimeError("unavailable")


class _Channel:
    def __init__(self, response: bool | BaseException) -> None:
        self.response = response

    def send(self, _subject, _body):
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class _Reader:
    def list_inventory(self) -> TaskInventory:
        return TaskInventory(items=(), complete=True, fresh=True)


@pytest.mark.parametrize(
    ("channel", "setup_error", "expected"),
    [
        (None, None, "not_configured"),
        (_Channel(True), None, "sent"),
        (_Channel(False), None, "delivery_failed"),
        (_Channel(RuntimeError("send failed")), None, "alert_error"),
        (None, ValueError("incomplete configuration"), "alert_error"),
    ],
)
def test_refresh_contains_all_operational_alert_outcomes(
    session, channel, setup_error, expected
) -> None:
    now = datetime.now(UTC)
    entity = TopicEntity(
        canonical_title="Candidate",
        canonical_url="https://example.test/candidate",
        category="engineering",
        earliest_seen=now,
        source_count=1,
        rank_position=0,
    )
    session.add(entity)
    session.flush()
    session.add(
        FeedItem(
            source="test",
            source_id="source",
            url="https://example.test/source",
            title="Source",
            content="evidence",
            published_at=now,
            topic_entity_id=entity.id,
        )
    )
    session.commit()
    result = run_backlog_mining(
        session,
        config=PipelineConfig(
            category_to_target_template={"engineering": TargetTemplate("owner/repo", "beads", "v1")}
        ),
        judge=_ExplodingJudge(),
        channel=channel,
        channel_setup_error=setup_error,
        task_store_reader=_Reader(),
    )
    assert isinstance(result.error, RuntimeError)
    assert result.alert_status == expected
    run = session.query(BacklogMiningRun).one()
    assert run.status == "failed"
    assert run.error_payload["alert_status"] == expected


def test_refresh_preserves_smtp_absence_and_setup_error_distinction(session) -> None:
    now = datetime.now(UTC)
    entity = TopicEntity(
        canonical_title="Candidate",
        canonical_url="https://example.test/candidate",
        category="engineering",
        earliest_seen=now,
        source_count=1,
        rank_position=0,
    )
    session.add(entity)
    session.flush()
    session.add(
        FeedItem(
            source="test",
            source_id="source-distinction",
            url="https://example.test/source",
            title="Source",
            content="evidence",
            published_at=now,
            topic_entity_id=entity.id,
        )
    )
    session.commit()
    config = PipelineConfig(
        category_to_target_template={"engineering": TargetTemplate("owner/repo", "beads", "v1")}
    )

    absent = run_backlog_mining(
        session,
        config=config,
        judge=_ExplodingJudge(),
        channel=None,
        channel_setup_error=None,
        task_store_reader=_Reader(),
    )
    setup_error = run_backlog_mining(
        session,
        config=config,
        judge=_ExplodingJudge(),
        channel=None,
        channel_setup_error=ValueError("incomplete SMTP operational-alert configuration"),
        task_store_reader=_Reader(),
    )

    runs = session.query(BacklogMiningRun).order_by(BacklogMiningRun.id).all()
    assert absent.alert_status == "not_configured"
    assert setup_error.alert_status == "alert_error"
    assert runs[0].error_payload["channel_setup_error"] is None
    assert "incomplete SMTP" in runs[1].error_payload["channel_setup_error"]


def test_mining_alert_when_another_run_starts_then_updates_the_failed_run_only(session) -> None:
    first = BacklogMiningRun(status="failed", config_snapshot={}, counts={}, error_payload={})
    second = BacklogMiningRun(status="running", config_snapshot={}, counts={})
    session.add_all((first, second))
    session.commit()

    _record_mining_alert(
        session,
        RuntimeError("first failed"),
        run_id=first.id,
        channel=None,
        channel_setup_error=None,
    )

    assert session.get(BacklogMiningRun, first.id).error_payload["alert_status"] == "not_configured"
    assert session.get(BacklogMiningRun, second.id).error_payload is None
