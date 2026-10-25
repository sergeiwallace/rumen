"""Chrome native-messaging host for validated Reading List snapshots."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from rumen.chrome_reading_list import SnapshotValidationError, validate_native_message

MAX_NATIVE_MESSAGE_BYTES = 1_048_576


class NativeMessageError(ValueError):
    """A native-messaging frame is malformed or exceeds the supported bound."""


def read_native_message(stream: BinaryIO) -> dict[str, Any] | None:
    """Read one Chrome native-messaging frame, returning ``None`` at clean EOF."""
    header = stream.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise NativeMessageError("incomplete native-messaging length prefix")
    length = struct.unpack("<I", header)[0]
    if length > MAX_NATIVE_MESSAGE_BYTES:
        raise NativeMessageError("native message exceeds maximum size")
    payload = stream.read(length)
    if len(payload) != length:
        raise NativeMessageError("incomplete native-messaging payload")
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeMessageError("native message must contain UTF-8 JSON") from exc
    if not isinstance(message, dict):
        raise NativeMessageError("native message must be a JSON object")
    return message


def write_native_message(stream: BinaryIO, response: dict[str, Any]) -> None:
    """Write one Chrome native-messaging response frame."""
    payload = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def accept_message(
    message: object,
    inbox_path: Path,
    *,
    replace: Callable[[str, str], None] = os.replace,
) -> dict[str, Any]:
    """Validate and atomically install a snapshot, acknowledging only after replacement."""
    snapshot = validate_native_message(message)
    _atomically_write_snapshot(inbox_path, snapshot, replace=replace)
    return {"ok": True, "snapshotId": snapshot["snapshotId"]}


def _atomically_write_snapshot(
    inbox_path: Path,
    snapshot: dict[str, Any],
    *,
    replace: Callable[[str, str], None],
) -> None:
    inbox_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{inbox_path.name}.", dir=inbox_path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(
                json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        replace(temporary_path, str(inbox_path))
        _fsync_directory(inbox_path.parent)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(
    inbox_path: Path,
    stdin: BinaryIO,
    stdout: BinaryIO,
    *,
    accept: Callable[[object, Path], dict[str, Any]] = accept_message,
) -> int:
    """Serve native messages until Chrome closes stdin."""
    while True:
        try:
            message = read_native_message(stdin)
        except NativeMessageError as exc:
            write_native_message(stdout, {"ok": False, "error": str(exc)})
            return 1
        if message is None:
            return 0
        try:
            response = accept(message, inbox_path)
        except (OSError, SnapshotValidationError) as exc:
            response = {"ok": False, "error": str(exc)}
        write_native_message(stdout, response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", required=True, type=Path, help="private snapshot inbox path")
    args = parser.parse_args()
    return run(args.inbox, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
