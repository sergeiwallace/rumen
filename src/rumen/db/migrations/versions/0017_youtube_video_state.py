"""YouTube playlist operational lifecycle state.

Revision ID: 0017
Revises: 0016
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "youtube_video_states",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("feed_item_id", sa.Integer(), sa.ForeignKey("feed_items.id"), nullable=False),
        sa.Column("playlist_id", sa.Text(), nullable=False),
        sa.Column("video_id", sa.Text(), nullable=False),
        sa.Column("membership_status", sa.Text(), nullable=False, server_default="unknown"),
        sa.Column("last_seen_in_playlist_at", sa.DateTime(timezone=True)),
        sa.Column("metadata_refreshed_at", sa.DateTime(timezone=True)),
        sa.Column("metadata_refresh_error", sa.Text()),
        sa.Column("transcript_status", sa.Text(), nullable=False, server_default="disabled"),
        sa.Column("transcript_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("transcript_next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("transcript_provider", sa.Text()),
        sa.Column("transcript_language", sa.Text()),
        sa.Column("transcript_kind", sa.Text()),
        sa.Column("transcript_fetched_at", sa.DateTime(timezone=True)),
        sa.Column("transcript_feed_item_id", sa.Integer(), sa.ForeignKey("feed_items.id")),
        sa.Column("last_error_code", sa.Text()),
        sa.UniqueConstraint("feed_item_id", name="uq_youtube_video_states_feed_item"),
        sa.UniqueConstraint(
            "playlist_id", "video_id", name="uq_youtube_video_states_playlist_video"
        ),
        sa.UniqueConstraint(
            "transcript_feed_item_id", name="uq_youtube_video_states_transcript_feed_item"
        ),
    )
    op.create_index(
        "idx_youtube_video_states_transcript_due",
        "youtube_video_states",
        ["transcript_status", "transcript_next_attempt_at"],
    )
    op.create_index(
        "idx_youtube_video_states_metadata_refreshed",
        "youtube_video_states",
        ["metadata_refreshed_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_youtube_video_states_metadata_refreshed", table_name="youtube_video_states")
    op.drop_index("idx_youtube_video_states_transcript_due", table_name="youtube_video_states")
    op.drop_table("youtube_video_states")
