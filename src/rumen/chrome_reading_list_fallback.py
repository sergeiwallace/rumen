"""Synthetic-only feasibility gate for a future Chrome Reading List fallback.

This deliberately does not open LevelDB, locate a Chrome profile, or expose a source adapter.
It inspects only the sanitized protobuf bytes stored in the fixture inventory so a later decoder
has explicit evidence and failure requirements before it can be proposed for human admission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

_DATA_PREFIX = "reading_list-dt-"
_STATUS_NAMES = {0: "UNREAD", 1: "READ", 2: "UNSEEN"}
_KNOWN_FIELDS = frozenset({1, 2, 3, 4, 5, 6})
_PINNED_PROTO_SOURCES = {
    120: ("120.0.6099.109", "1dedb75d97e7964a37cde05306c41006edc532aa5629010b8b8fc85c3b7a3e01"),
    130: ("130.0.6723.91", "bb9ba92cd60203074adb118f9eb13e3ac54cd6e9de1403a1cb44e9df5071800c"),
    140: ("140.0.7339.80", "bb9ba92cd60203074adb118f9eb13e3ac54cd6e9de1403a1cb44e9df5071800c"),
}


class FixtureValidationError(ValueError):
    """A sanitized fixture does not prove the required fallback invariant."""


@dataclass(frozen=True, slots=True)
class FixtureGateResult:
    """The non-activation verdict for one source-pinned milestone fixture."""

    milestone: int
    feasible: bool
    reason: str | None
    urls: tuple[str, ...] = ()
    updated_urls: tuple[str, ...] = ()
    deleted_urls: tuple[str, ...] = ()
    unicode_url_count: int = 0
    unknown_fields_preserved: bool = False
    corruption_rejected: bool = False


@dataclass(frozen=True, slots=True)
class OfflineFallbackGateReport:
    """Gate evidence; ``fallback_enabled`` is intentionally never true in this task."""

    fixtures: tuple[FixtureGateResult, ...]
    overall_feasible: bool
    fallback_enabled: bool
    live_leveldb_opened: bool


def run_offline_fallback_gate(inventory_path: str | Path) -> OfflineFallbackGateReport:
    """Validate a synthetic fixture inventory without ever opening a LevelDB database."""
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    if inventory.get("format") != "rumen.chrome_reading_list.offline_fixture_inventory.v1":
        raise FixtureValidationError("unsupported fixture inventory format")
    if inventory.get("fixture_origin") != "synthetic-protobuf-records-only":
        raise FixtureValidationError("fixture inventory is not synthetic-only")
    if inventory.get("live_leveldb_opened") is not False:
        raise FixtureValidationError(
            "fixture inventory must attest that no live LevelDB was opened"
        )

    fixtures = inventory.get("fixtures")
    milestones = inventory.get("supported_milestones")
    if not isinstance(fixtures, list) or not isinstance(milestones, list):
        raise FixtureValidationError(
            "fixture inventory must define fixtures and supported milestones"
        )

    results = tuple(_validate_fixture(fixture) for fixture in fixtures)
    expected_milestones = tuple(milestones)
    if tuple(result.milestone for result in results) != expected_milestones:
        raise FixtureValidationError("fixture order must exactly match supported milestones")
    return OfflineFallbackGateReport(
        fixtures=results,
        overall_feasible=all(result.feasible for result in results),
        # T-05 establishes evidence only. Activation requires the plan's separate human gate.
        fallback_enabled=False,
        live_leveldb_opened=False,
    )


def _validate_fixture(fixture: object) -> FixtureGateResult:
    if not isinstance(fixture, dict) or not isinstance(fixture.get("milestone"), int):
        raise FixtureValidationError("each fixture must have an integer milestone")
    milestone = fixture["milestone"]
    if fixture.get("requires_live_leveldb") is True:
        return FixtureGateResult(milestone, False, "fixture requires a live LevelDB")
    if fixture.get("storage_variant") != "default":
        return FixtureGateResult(
            milestone, False, f"unsupported storage variant: {fixture.get('storage_variant')}"
        )
    if _PINNED_PROTO_SOURCES.get(milestone) != (
        fixture.get("source_ref"),
        fixture.get("chromium_proto_sha256"),
    ):
        return FixtureGateResult(milestone, False, "fixture is not source-pinned")

    try:
        return _inspect_default_storage_fixture(milestone, fixture)
    except FixtureValidationError as exc:
        return FixtureGateResult(milestone, False, str(exc))


def _inspect_default_storage_fixture(milestone: int, fixture: dict) -> FixtureGateResult:
    records = fixture.get("records")
    deleted_urls = fixture.get("deleted_urls")
    if not isinstance(records, list) or not isinstance(deleted_urls, list):
        raise FixtureValidationError("fixture records and deleted URLs must be lists")

    data_keys: set[str] = set()
    urls: list[str] = []
    updated_urls: list[str] = []
    unicode_url_count = 0
    unknown_fields_preserved = True
    corruption_rejected = True

    for record in records:
        if not isinstance(record, dict):
            raise FixtureValidationError("fixture record must be an object")
        key = record.get("key")
        if not isinstance(key, str):
            raise FixtureValidationError("fixture record key must be a string")
        if not key.startswith(_DATA_PREFIX):
            continue
        if key in data_keys:
            raise FixtureValidationError(f"duplicate live data key: {key}")
        data_keys.add(key)

        value = _decode_hex(record.get("value_hex"), "value")
        fields, unknown_fields = _scan_protobuf(value)
        url = _validate_entry(key, fields, record.get("expected"))
        urls.append(url)
        expected = record["expected"]
        unknown_fields_preserved = unknown_fields_preserved and _unknown_fields_match(
            unknown_fields, expected["unknown_field_hex"]
        )
        previous_update = expected.get("previous_update_time_us")
        if previous_update is not None:
            if not isinstance(previous_update, int) or previous_update >= fields[5]:
                raise FixtureValidationError(f"update is not newer for {url}")
            updated_urls.append(url)
        corrupt_value = _decode_hex(record.get("corrupt_value_hex"), "corrupt value")
        if not corrupt_value:
            corrupt_value = value[:-1]
        corruption_rejected = corruption_rejected and _is_corruption_rejected(corrupt_value)
        if _contains_unicode_url(url):
            unicode_url_count += 1

    if not data_keys:
        raise FixtureValidationError("fixture has no live reading_list-dt- records")
    deleted = _validate_deletions(deleted_urls, data_keys)
    if not unknown_fields_preserved:
        raise FixtureValidationError("unknown protobuf field bytes are not retained by the fixture")
    if not corruption_rejected:
        raise FixtureValidationError("corrupt protobuf fixture was accepted")
    if unicode_url_count != len(urls):
        raise FixtureValidationError("fixture does not exercise a percent-encoded Unicode URL")
    return FixtureGateResult(
        milestone=milestone,
        feasible=True,
        reason=None,
        urls=tuple(urls),
        updated_urls=tuple(updated_urls),
        deleted_urls=deleted,
        unicode_url_count=unicode_url_count,
        unknown_fields_preserved=unknown_fields_preserved,
        corruption_rejected=corruption_rejected,
    )


def _validate_entry(key: str, fields: dict[int, int | str], expected: object) -> str:
    if not isinstance(expected, dict):
        raise FixtureValidationError(f"live data record {key} has no expected mapping")
    required = {1, 2, 3, 4, 5, 6}
    if set(fields) != required:
        raise FixtureValidationError(f"record has unexpected required fields: {key}")
    entry_id, title, url = fields[1], fields[2], fields[3]
    if not all(isinstance(value, str) for value in (entry_id, title, url)):
        raise FixtureValidationError(f"record text fields have an invalid wire type: {key}")
    if key != f"{_DATA_PREFIX}{url}" or entry_id != url:
        raise FixtureValidationError(f"key, entry ID, and URL do not match: {key}")
    if title != expected.get("title"):
        raise FixtureValidationError(f"title mapping failed: {key}")
    if fields[4] != expected.get("creation_time_us") or fields[5] != expected.get("update_time_us"):
        raise FixtureValidationError(f"timestamp mapping failed: {key}")
    if not isinstance(fields[4], int) or not isinstance(fields[5], int) or fields[5] < fields[4]:
        raise FixtureValidationError(f"timestamps are invalid: {key}")
    if _STATUS_NAMES.get(fields[6]) != expected.get("status"):
        raise FixtureValidationError(f"status mapping failed: {key}")
    return url


def _validate_deletions(deleted_urls: list[object], data_keys: set[str]) -> tuple[str, ...]:
    if not deleted_urls or not all(isinstance(url, str) and url for url in deleted_urls):
        raise FixtureValidationError("fixture must include a deleted URL")
    for url in deleted_urls:
        if f"{_DATA_PREFIX}{url}" in data_keys:
            raise FixtureValidationError(f"deleted URL remains live: {url}")
    return tuple(deleted_urls)


def _contains_unicode_url(url: str) -> bool:
    parsed = urlsplit(url)
    return bool(parsed.fragment) and any(ord(character) > 127 for character in unquote(url))


def _unknown_fields_match(unknown_fields: list[bytes], expected_hex: object) -> bool:
    if not isinstance(expected_hex, str):
        return False
    return bytes.fromhex(expected_hex) in unknown_fields


def _is_corruption_rejected(value: bytes) -> bool:
    try:
        _scan_protobuf(value)
    except FixtureValidationError:
        return True
    return False


def _decode_hex(value: object, label: str) -> bytes:
    if not isinstance(value, str):
        if label == "corrupt value":
            return b""
        raise FixtureValidationError(f"{label} must be hexadecimal text")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise FixtureValidationError(f"{label} is not valid hexadecimal") from exc


def _scan_protobuf(value: bytes) -> tuple[dict[int, int | str], list[bytes]]:
    """Read enough protobuf wire structure to validate fixture evidence, not a decoder API."""
    fields: dict[int, int | str] = {}
    unknown_fields: list[bytes] = []
    index = 0
    while index < len(value):
        field_start = index
        tag, index = _read_varint(value, index)
        field_number, wire_type = tag >> 3, tag & 0b111
        if field_number == 0 or wire_type in {3, 4}:
            raise FixtureValidationError("invalid protobuf field tag")
        payload, index = _read_wire_value(value, index, wire_type)
        if field_number not in _KNOWN_FIELDS:
            unknown_fields.append(value[field_start:index])
            continue
        if field_number in fields:
            raise FixtureValidationError(f"duplicate protobuf field {field_number}")
        fields[field_number] = _decode_known_field(field_number, wire_type, payload)
    return fields, unknown_fields


def _read_varint(value: bytes, index: int) -> tuple[int, int]:
    result = 0
    for shift in range(0, 70, 7):
        if index >= len(value):
            raise FixtureValidationError("truncated protobuf varint")
        byte = value[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, index
    raise FixtureValidationError("protobuf varint exceeds 64 bits")


def _read_wire_value(value: bytes, index: int, wire_type: int) -> tuple[int | bytes, int]:
    if wire_type == 0:
        return _read_varint(value, index)
    if wire_type == 1:
        end = index + 8
    elif wire_type == 2:
        size, index = _read_varint(value, index)
        end = index + size
    elif wire_type == 5:
        end = index + 4
    else:
        raise FixtureValidationError(f"unsupported protobuf wire type {wire_type}")
    if end > len(value):
        raise FixtureValidationError("truncated protobuf value")
    return value[index:end], end


def _decode_known_field(field_number: int, wire_type: int, payload: int | bytes) -> int | str:
    if field_number in {1, 2, 3}:
        if wire_type != 2 or not isinstance(payload, bytes):
            raise FixtureValidationError(f"protobuf field {field_number} must be text")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FixtureValidationError(f"protobuf field {field_number} is not UTF-8") from exc
    if wire_type != 0 or not isinstance(payload, int):
        raise FixtureValidationError(f"protobuf field {field_number} must be a varint")
    return payload
