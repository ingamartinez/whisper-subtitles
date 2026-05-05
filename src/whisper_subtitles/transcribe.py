"""Transcription layer: WAV -> word-level timestamps via whisper.cpp."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

VAD_MODEL_FILENAME = "ggml-silero-v5.1.2.bin"


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Transcription:
    language: str
    words: list[Word]


def transcribe(
    wav_path: Path,
    model_path: Path,
    language: str | None = None,
    vad: bool = True,
) -> Transcription:
    """Run whisper.cpp on a WAV file and return word-level transcription.

    `language` is an ISO code like 'es' or 'en'. None means auto-detect.
    `vad` enables Voice Activity Detection pre-segmentation (recommended
    — much more accurate timestamps, especially around silence regions).
    """
    with tempfile.TemporaryDirectory(prefix="whisper-subtitles-") as tmp:
        out_prefix = Path(tmp) / "out"
        cmd = [
            "whisper-cli",
            "-m", str(model_path),
            "-f", str(wav_path),
            "-l", language or "auto",
            "-ojf",
            "-ml", "1",
            "-sow",
            "-sns",
            "-np",
            "-of", str(out_prefix),
        ]

        if vad:
            vad_model = model_path.parent / VAD_MODEL_FILENAME
            if not vad_model.exists():
                raise RuntimeError(
                    f"VAD enabled but model not found at {vad_model}. "
                    f"Run scripts/setup.sh to download it, or pass --no-vad."
                )
            cmd.extend(["--vad", "--vad-model", str(vad_model)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"whisper-cli failed: {result.stderr.strip()}")

        json_path = out_prefix.with_suffix(".json")
        data = json.loads(json_path.read_text(encoding="utf-8"))

    detected_language = data.get("result", {}).get("language", language or "unknown")
    words: list[Word] = []
    for entry in data.get("transcription", []):
        text = entry.get("text", "").strip()
        if not text:
            continue
        offsets = entry["offsets"]
        words.append(
            Word(
                text=text,
                start=offsets["from"] / 1000.0,
                end=offsets["to"] / 1000.0,
            )
        )

    return Transcription(language=detected_language, words=words)
