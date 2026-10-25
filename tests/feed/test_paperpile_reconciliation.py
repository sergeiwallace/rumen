from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rumen.adapters.base import RawItem, SourceChange
from rumen.ingest import run_ingestion
from rumen.models import Base, FeedItem


class _Changes:
    def __init__(self, changes):
        self.changes = changes

    def source_name(self):
        return "paperpile:personal"

    def fetch_changes(self, since):
        return [change for change in self.changes if since is None or change.occurred_at > since]


def _change(op: str, at: datetime, revision: str, title: str = "Paper") -> SourceChange:
    item = (
        None
        if op == "tombstone"
        else RawItem("paperpile:personal", "work", "paperpile://personal/work", title, at)
    )
    return SourceChange(op, "work", at, revision, item)


@pytest.fixture
def change_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_changes_when_upsert_edit_tombstone_restore_then_converges_on_one_active_projection(
    change_session,
) -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    adapter = _Changes(
        [
            _change("upsert", start, "one"),
            _change("upsert", start + timedelta(seconds=1), "two", "Edited"),
            _change("tombstone", start + timedelta(seconds=2), "two"),
            _change("upsert", start + timedelta(seconds=3), "three", "Restored"),
        ]
    )
    run = run_ingestion(change_session, adapter)
    row = change_session.scalar(select(FeedItem))
    assert run.status == "completed"
    assert (
        row.title == "Restored"
        and row.source_deleted_at is None
        and row.source_revision_digest == "three"
    )
    assert change_session.query(FeedItem).count() == 1
