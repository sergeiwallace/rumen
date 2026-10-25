from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rumen.adapters.reddit_subreddits import RedditSubredditSettings
from rumen.models import DroppedItem, FeedItem, SourceMaintenanceRun
from rumen.reddit_retention import RedditRetentionMaintainer


class Gateway:
    def __init__(self, returned=(), error: Exception | None = None):
        self.returned, self.error = returned, error

    def listing(self, subreddit, mode, *, limit):
        return iter(())

    def revalidate(self, fullnames):
        if self.error:
            raise self.error
        return iter(self.returned)


def _maintainer(gateway: Gateway) -> RedditRetentionMaintainer:
    return RedditRetentionMaintainer(
        RedditSubredditSettings(
            "ai", ("Test",), ("submissions",), "ID", "SECRET", "ua", "approval"
        ),
        gateway,
    )


def _item(source_id: str, published_at: datetime) -> FeedItem:
    return FeedItem(
        source="reddit:ai",
        source_id=source_id,
        url="https://www.reddit.com/r/Test/comments/a",
        title="secret title canary",
        content="secret body canary",
        published_at=published_at,
    )


def test_given_expired_reddit_item_when_maintenance_runs_then_purges_without_provider(
    session,
) -> None:
    item = _item("t3_old", datetime.now(UTC) - timedelta(hours=49))
    session.add_all(
        [item, DroppedItem(source="reddit:ai", source_id="t3_old", title="canary", drop_reason="x")]
    )
    session.commit()
    run = _maintainer(Gateway()).maintain(session)
    assert run.status == "completed" and run.items_purged == 1
    assert session.get(FeedItem, item.id) is None
    assert session.query(DroppedItem).filter_by(source="reddit:ai").count() == 0
    assert "canary" not in str(session.get(SourceMaintenanceRun, run.id).reason_counts)


def test_given_missing_revalidation_item_when_maintenance_runs_then_purges_and_is_idempotent(
    session,
) -> None:
    item = _item("t3_missing", datetime.now(UTC) - timedelta(hours=1))
    session.add(item)
    session.commit()
    maintainer = _maintainer(Gateway())
    assert maintainer.maintain(session).items_purged == 1
    assert maintainer.maintain(session).items_purged == 0


def test_given_provider_outage_after_expiry_when_maintenance_runs_then_records_failed_without_restore(
    session,
) -> None:
    item = _item("t3_old", datetime.now(UTC) - timedelta(hours=49))
    # A still-retained (non-expired) item is required so the revalidation call actually
    # runs after the expiry purge -- an all-expired queue never reaches the gateway.
    live = _item("t3_live", datetime.now(UTC) - timedelta(hours=1))
    session.add_all([item, live])
    session.commit()
    run = _maintainer(Gateway(error=RuntimeError("secret-canary"))).maintain(session)
    assert run.status == "failed" and run.items_purged == 1
    assert session.get(FeedItem, item.id) is None
    assert session.get(FeedItem, live.id) is not None
    assert "secret-canary" not in (run.error_message or "")


def test_given_termination_when_targeting_profile_then_preserves_other_reddit_namespace(
    session,
) -> None:
    target = _item("t3_target", datetime.now(UTC))
    saved = FeedItem(
        source="reddit_saved",
        source_id="t3_saved",
        url="u",
        title="saved",
        published_at=datetime.now(UTC),
    )
    session.add_all([target, saved])
    session.commit()
    assert _maintainer(Gateway()).terminate(session) == 1
    assert session.get(FeedItem, target.id) is None
    assert session.get(FeedItem, saved.id) is not None
