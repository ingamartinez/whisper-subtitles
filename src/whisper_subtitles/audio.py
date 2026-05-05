"""Audio extraction layer: video file -> WAV 16kHz mono (whisper.cpp input format)."""

from __future__ import annotations

import subprocess
from pathlib import Path

WHISPER_SAMPLE_RATE = 16_000


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract audio from a video into a 16kHz mono PCM WAV file.

    Returns the path to the generated WAV. Raises RuntimeError if ffmpeg fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-ar", str(WHISPER_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting audio from {video_path}: {result.stderr.strip()}"
        )

    return wav_path
