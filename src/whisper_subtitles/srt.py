"""SRT serialization layer: list of Cue -> .srt file on disk."""

from __future__ import annotations

from pathlib import Path

from whisper_subtitles.chunker import Cue


def write_srt(cues: list[Cue], output_path: Path) -> None:
    """Write a list of cues to disk in SubRip (.srt) format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    for cue in cues:
        blocks.append(
            f"{cue.index}\n"
            f"{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n"
            f"{cue.text}\n\n"
        )

    output_path.write_text("".join(blocks), encoding="utf-8")


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SubRip timestamp `HH:MM:SS,mmm` (comma decimal separator)."""
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
