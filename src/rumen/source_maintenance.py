"""Narrow lifecycle-maintenance protocol used by pipeline orchestration."""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from rumen.models import SourceMaintenanceRun


class SourceMaintainer(Protocol):
    def source_name(self) -> str: ...

    def maintain(self, session: Session) -> SourceMaintenanceRun: ...
