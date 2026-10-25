"""Rumen's deployment-neutral operational notification port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class NotificationChannel(Protocol):
    def send(self, subject: str, body: str) -> bool:
        """Attempt delivery; False is a contained delivery failure."""
