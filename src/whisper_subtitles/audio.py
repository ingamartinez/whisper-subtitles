"""Audio extraction layer: video -> WAV 16kHz mono with dynamic loudness normalization."""

from __future__ import annotations

import subprocess
from pathlib import Path

WHISPER_SAMPLE_RATE = 16_000

# dynaudnorm settings:
#   f=150  → 150ms analysis frame (responsive to short utterances)
#   g=15   → 15-frame Gaussian smoothing window (~2.25s context)
#   p=0.95 → target peak amplitude 0.95 (-0.5dBFS)
#   m=12   → max gain factor (lifts very quiet speech up to 12x)
#   r=0.0  → RMS-based gain (off; uses peak)
DYNAUDNORM_FILTER = "dynaudnorm=f=150:g=15:p=0.95:m=12"


def extract_audio(
    video_path: Path,
    output_dir: Path,
    normalize: bool = False,
) -> Path:
    """Extract audio from a video to 16kHz mono PCM WAV.

    With `normalize=True` (default), applies ffmpeg dynaudnorm to lift
    quiet speakers and contain loud ones — critical when the recording
    has multiple voices at different volumes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
    ]
    if normalize:
        cmd.extend(["-af", DYNAUDNORM_FILTER])
    cmd.extend([
        "-ar", str(WHISPER_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting audio from {video_path}: {result.stderr.strip()}"
        )

    return wav_path
