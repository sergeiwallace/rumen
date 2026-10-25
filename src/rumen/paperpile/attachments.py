"""Safe attachment manifests; filenames are never identity or join evidence."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path


class AttachmentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Attachment:
    relative_path: str
    sha256: str
    size: int
    kind: str
    relationship_provenance: str


def safe_attachment(root: Path, locator: str, *, provenance: str) -> Attachment:
    root = root.resolve()
    path = (root / locator).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise AttachmentError("attachment locator escapes configured root") from exc
    if not path.is_file():
        raise AttachmentError("attachment locator is missing")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise AttachmentError("attachment changed during hashing")
    kind = (
        "pdf" if path.suffix.lower() == ".pdf" else (mimetypes.guess_type(path.name)[0] or "other")
    )
    return Attachment(str(relative), digest.hexdigest(), before.st_size, kind, provenance)


def manifest(root: Path) -> dict[str, tuple[int, int]]:
    root = root.resolve()
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve().is_relative_to(root)
    }
