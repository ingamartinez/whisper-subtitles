"""Built-in chunking presets — different subtitle styles for different use cases."""

from __future__ import annotations

from whisper_subtitles.chunker import ChunkRules

TRADITIONAL = ChunkRules(
    max_chars_per_line=42,
    max_lines_per_cue=2,
    max_duration_seconds=6.0,
    min_duration_seconds=1.0,
    gap_threshold_seconds=0.5,
    prefer_sentence_breaks=True,
)

SOCIAL = ChunkRules(
    max_chars_per_line=30,
    max_chars_overflow=4,
    max_lines_per_cue=2,
    max_duration_seconds=3.0,
    min_duration_seconds=0.6,
    gap_threshold_seconds=0.4,
    prefer_sentence_breaks=True,
    orphan_max_words=2,
)

KARAOKE = ChunkRules(
    max_chars_per_line=1,
    max_lines_per_cue=1,
    max_duration_seconds=2.0,
    min_duration_seconds=0.2,
    gap_threshold_seconds=10.0,
    prefer_sentence_breaks=False,
)

DILAN = ChunkRules(
    max_chars_per_line=13,
    max_lines_per_cue=1,
    max_words_per_cue=3,
    max_duration_seconds=2.0,
    min_duration_seconds=0.2,
    gap_threshold_seconds=0.3,
    prefer_sentence_breaks=True,
)

PRESETS: dict[str, ChunkRules] = {
    "traditional": TRADITIONAL,
    "social": SOCIAL,
    "karaoke": KARAOKE,
    "dilan": DILAN,
}
