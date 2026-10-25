from pathlib import Path

import pytest

from rumen.paperpile.attachments import AttachmentError, safe_attachment


def test_attachment_when_locator_is_exported_under_root_then_preserves_relative_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "drive"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"pdf")
    attachment = safe_attachment(root, "paper.pdf", provenance="json:attachments")
    assert attachment.relative_path == "paper.pdf"
    assert attachment.kind == "pdf"
    assert attachment.relationship_provenance == "json:attachments"


def test_attachment_when_locator_escapes_root_then_rejects_it(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    root.mkdir()
    (tmp_path / "outside.pdf").write_bytes(b"private")
    with pytest.raises(AttachmentError, match="escapes"):
        safe_attachment(root, "../outside.pdf", provenance="json:attachments")
