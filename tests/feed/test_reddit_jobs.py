from __future__ import annotations

from datetime import UTC, datetime

from rumen.adapters.base import RawItem
from rumen.jobs import run_pipeline
from rumen.models import FeedItem, SourceMaintenanceRun


class Adapter:
    def source_name(self) -> str:
        return "reddit:ai"

    def fetch_new(self, since):
        return [RawItem("reddit:ai", "t3_one", "https://example.test", "title", datetime.now(UTC))]


class Maintainer:
    def __init__(self, status: str):
        self.status = status

    def source_name(self) -> str:
        return "reddit:ai"

    def maintain(self, session):
        run = SourceMaintenanceRun(
            source="reddit:ai",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            status=self.status,
        )
        session.add(run)
        session.commit()
        return run


def test_given_completed_maintenance_when_pipeline_runs_then_ingests_matching_adapter(
    session,
) -> None:
    result = run_pipeline(
        session, adapters=[Adapter()], source_maintainers=[Maintainer("completed")]
    )
    assert result.ok and session.query(FeedItem).filter_by(source="reddit:ai").count() == 1


def test_given_failed_maintenance_when_pipeline_runs_then_skips_only_matching_adapter(
    session,
) -> None:
    result = run_pipeline(session, adapters=[Adapter()], source_maintainers=[Maintainer("failed")])
    assert not result.ok and session.query(FeedItem).filter_by(source="reddit:ai").count() == 0
