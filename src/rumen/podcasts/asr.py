"""Safe local-ASR subprocess seam; it never invokes a shell or hosted service."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rumen.podcasts.transcripts import NormalizedTranscript, normalize_transcript


class AsrError(RuntimeError):
    """The local ASR command did not produce an acceptable transcript."""


@dataclass(frozen=True)
class AsrConfig:
    argv: list[str]
    tool_model_identifier: str
    timeout_seconds: float = 900
    output_mime_type: str = "text/vtt"
    output_byte_ceiling: int = 20 * 1024 * 1024

    def validate(self) -> None:
        if not self.argv or not all(isinstance(part, str) and part for part in self.argv):
            raise AsrError("ASR argv must be a non-empty string list")
        if "{input}" not in self.argv or "{output}" not in self.argv:
            raise AsrError("ASR argv must contain {input} and {output} arguments")
        if (
            not self.tool_model_identifier
            or self.timeout_seconds <= 0
            or self.output_byte_ceiling <= 0
        ):
            raise AsrError("ASR configuration has invalid limits or tool/model identifier")


def transcribe(audio: bytes, config: AsrConfig, *, suffix: str = ".audio") -> NormalizedTranscript:
    config.validate()
    with tempfile.TemporaryDirectory(prefix="rumen-podcast-asr-") as temp_dir:
        directory = Path(temp_dir)
        input_path = directory / f"input{suffix}"
        output_path = directory / "transcript.out"
        input_path.write_bytes(audio)
        argv = [
            part.replace("{input}", str(input_path)).replace("{output}", str(output_path))
            for part in config.argv
        ]
        try:
            completed = subprocess.run(
                argv, shell=False, capture_output=True, timeout=config.timeout_seconds, check=False
            )  # noqa: S603
        except subprocess.TimeoutExpired as exc:
            raise AsrError("local ASR timed out") from exc
        except OSError as exc:
            raise AsrError("local ASR could not start") from exc
        if completed.returncode != 0:
            raise AsrError(f"local ASR exited with status {completed.returncode}")
        try:
            output = output_path.read_bytes()
        except OSError as exc:
            raise AsrError("local ASR produced no output") from exc
        if len(output) > config.output_byte_ceiling:
            raise AsrError("local ASR output exceeds byte ceiling")
        try:
            return normalize_transcript(output, config.output_mime_type)
        except ValueError as exc:
            raise AsrError("local ASR output is invalid") from exc
