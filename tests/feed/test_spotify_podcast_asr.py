"""Local-ASR rights, process-boundary, cleanup, purge, and summary regressions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rumen.podcasts.asr import AsrConfig, AsrError, transcribe
from rumen.podcasts.cache import initialize, purge_import, transaction
from rumen.podcasts.http import HttpResponse, PublisherHttpClient
from rumen.podcasts.sync import synchronize

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "spotify_podcasts"
_HISTORY = _FIXTURES / "history.json"
_FEED = (_FIXTURES / "feed.xml").read_bytes()
_VTT = (_FIXTURES / "transcript.vtt").read_bytes()


def test_given_fake_local_executable_when_asr_runs_then_parses_output_and_removes_temp_files(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.txt"
    code = "from pathlib import Path; import sys; Path(sys.argv[2]).write_text(str(Path(sys.argv[1]).parent)); Path(sys.argv[3]).write_text('WEBVTT\\n\\n00:00:00.000 --> 00:00:01.000\\nhello')"
    config = AsrConfig(
        [sys.executable, "-c", code, "{input}", str(trace), "{output}"], "fake-whisper"
    )

    normalized = transcribe(b"audio", config)

    assert normalized.plain_text == "hello"
    assert not Path(trace.read_text()).exists()


@pytest.mark.parametrize(
    "config",
    [
        AsrConfig([], "tool"),
        AsrConfig(["tool", "{input}"], "tool"),
        AsrConfig(["tool", "{input}", "{output}"], ""),
    ],
)
def test_given_invalid_asr_argv_when_validating_then_rejects_before_execution(
    config: AsrConfig,
) -> None:
    with pytest.raises(AsrError):
        config.validate()


def test_given_asr_nonzero_or_invalid_output_when_running_then_keeps_no_generated_text() -> None:
    failing = AsrConfig(
        [sys.executable, "-c", "import sys; sys.exit(2)", "{input}", "{output}"], "fake"
    )
    with pytest.raises(AsrError, match="status 2"):
        transcribe(b"audio", failing)


def test_given_no_publisher_text_and_rights_disabled_when_syncing_then_never_fetches_audio(
    tmp_path: Path,
) -> None:
    feed = _FEED.replace(
        b'<podcast:transcript url="https://publisher.example/one.vtt" type="text/vtt" language="en"/>',
        b"",
    )
    urls: list[str] = []
    manifest = {
        "cache_path": str(tmp_path / "cache.db"),
        "exports": [str(_HISTORY)],
        "feeds": [{"show": "Test Show", "url": "https://publisher.example/feed"}],
    }
    client = PublisherHttpClient(
        resolver=lambda _host: ["8.8.8.8"],
        transport=lambda url, *_args: urls.append(url) or HttpResponse(200, {}, feed),
        request_interval_seconds=0,
    )
    result = synchronize(manifest, client=client)
    assert (result["skipped_rights"], urls) == (1, ["https://publisher.example/feed"])


def test_given_publisher_text_and_asr_allowed_when_syncing_then_does_not_fetch_audio(
    tmp_path: Path,
) -> None:
    urls: list[str] = []
    manifest = {
        "cache_path": str(tmp_path / "cache.db"),
        "exports": [str(_HISTORY)],
        "feeds": [
            {
                "show": "Test Show",
                "url": "https://publisher.example/feed",
                "allow_asr": True,
                "asr": {"argv": ["tool", "{input}", "{output}"], "tool_model_identifier": "tool"},
            }
        ],
    }
    client = PublisherHttpClient(
        resolver=lambda _host: ["8.8.8.8"],
        transport=lambda url, *_args: (
            urls.append(url) or HttpResponse(200, {}, _FEED if url.endswith("feed") else _VTT)
        ),
        request_interval_seconds=0,
    )
    result = synchronize(manifest, client=client)
    assert result["publisher_transcript"] == 1
    assert "https://publisher.example/audio-one.mp3" not in urls


def test_given_allowed_feed_without_publisher_text_when_syncing_then_runs_local_asr_only_on_enclosure(
    tmp_path: Path,
) -> None:
    feed = _FEED.replace(
        b'<podcast:transcript url="https://publisher.example/one.vtt" type="text/vtt" language="en"/>',
        b"",
    )
    code = "from pathlib import Path; import sys; Path(sys.argv[2]).write_text('WEBVTT\\n\\n00:00:00.000 --> 00:00:01.000\\nlocal text')"
    urls: list[str] = []
    manifest = {
        "cache_path": str(tmp_path / "cache.db"),
        "exports": [str(_HISTORY)],
        "feeds": [
            {
                "show": "Test Show",
                "url": "https://publisher.example/feed",
                "allow_asr": True,
                "asr": {
                    "argv": [sys.executable, "-c", code, "{input}", "{output}"],
                    "tool_model_identifier": "fake-whisper-model",
                },
            }
        ],
    }
    client = PublisherHttpClient(
        resolver=lambda _host: ["8.8.8.8"],
        transport=lambda url, *_args: (
            urls.append(url) or HttpResponse(200, {}, feed if url.endswith("feed") else b"audio")
        ),
        request_interval_seconds=0,
    )

    result = synchronize(manifest, client=client)

    assert result["asr"] == 1
    assert urls == ["https://publisher.example/feed", "https://publisher.example/audio-one.mp3"]
    conn = initialize(manifest["cache_path"])
    row = conn.execute(
        "SELECT source_class, generator_identifier FROM podcast_transcripts"
    ).fetchone()
    assert tuple(row) == ("generated-local-asr", "fake-whisper-model")
    conn.close()


def test_given_import_purge_when_last_event_removed_then_reports_orphan_and_cascades(
    tmp_path: Path,
) -> None:
    # Build a ready cache through the same fixture-backed sync path.
    manifest = {
        "cache_path": str(tmp_path / "cache.db"),
        "exports": [str(_HISTORY)],
        "feeds": [{"show": "Test Show", "url": "https://publisher.example/feed"}],
    }
    client = PublisherHttpClient(
        resolver=lambda _host: ["8.8.8.8"],
        transport=lambda url, *_args: HttpResponse(
            200, {}, _FEED if url.endswith("feed") else b"text"
        ),
        request_interval_seconds=0,
    )
    synchronize(manifest, client=client)
    conn = initialize(manifest["cache_path"])
    digest = conn.execute("SELECT file_sha256 FROM spotify_podcast_imports").fetchone()[0]
    with transaction(conn):
        orphaned = purge_import(conn, digest)
    assert len(orphaned) == 1
    assert conn.execute("SELECT COUNT(*) FROM podcast_transcripts").fetchone()[0] == 0
    conn.close()
