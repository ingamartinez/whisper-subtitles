"""Chunking layer: word stream -> SRT cues using configurable rules.

Two-pass algorithm when `prefer_sentence_breaks` is enabled:
  1. Split words at sentence boundaries (. ? !).
  2. For sentences that don't fit one cue, split at clause boundaries (, ; :).
  3. Anything still too big falls back to greedy.

This is the seam where Phase 2 will plug in per-client rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from whisper_subtitles.transcribe import Word

OVERLAP_SAFETY_GAP = 0.05
SENTENCE_END_PUNCT = (".", "?", "!")
CLAUSE_END_PUNCT = (",", ";", ":")


@dataclass(frozen=True, slots=True)
class ChunkRules:
    max_chars_per_line: int = 42
    max_chars_overflow: int = 0
    max_lines_per_cue: int = 2
    max_duration_seconds: float = 6.0
    min_duration_seconds: float = 1.0
    gap_threshold_seconds: float = 0.5
    prefer_sentence_breaks: bool = True
    orphan_max_words: int = 0


@dataclass(frozen=True, slots=True)
class Cue:
    index: int
    start: float
    end: float
    text: str


DEFAULT_RULES = ChunkRules()


def chunk_words(words: list[Word], rules: ChunkRules = DEFAULT_RULES) -> list[Cue]:
    """Group words into SRT cues respecting the given chunking rules."""
    if not words:
        return []

    if rules.prefer_sentence_breaks:
        groups = _chunk_punct_aware(words, rules)
    else:
        groups = _greedy_split(words, rules)

    groups = _absorb_orphans(groups, rules)

    cues = [
        Cue(
            index=i,
            start=group[0].start,
            end=group[-1].end,
            text=_format_text(group, rules),
        )
        for i, group in enumerate(groups, start=1)
    ]

    return _enforce_min_duration(cues, rules)


def _absorb_orphans(groups: list[list[Word]], rules: ChunkRules) -> list[list[Word]]:
    """Merge tiny trailing groups into the previous one using the overflow margin."""
    if rules.orphan_max_words <= 0 or rules.max_chars_overflow <= 0:
        return groups

    out: list[list[Word]] = []
    for group in groups:
        if not out or len(group) > rules.orphan_max_words:
            out.append(group)
            continue

        candidate = out[-1] + group
        duration = candidate[-1].end - candidate[0].start
        if duration > rules.max_duration_seconds:
            out.append(group)
            continue

        if _layout_lines(candidate, rules, allow_overflow=True) is None:
            out.append(group)
            continue

        out[-1] = candidate

    return out


def _chunk_punct_aware(words: list[Word], rules: ChunkRules) -> list[list[Word]]:
    out: list[list[Word]] = []
    for sentence in _split_at(words, SENTENCE_END_PUNCT):
        if _fits_one_cue(sentence, rules):
            out.append(sentence)
            continue
        out.extend(_combine_clauses(sentence, rules))
    return out


def _combine_clauses(sentence_words: list[Word], rules: ChunkRules) -> list[list[Word]]:
    """Greedily combine adjacent clauses while they still fit one cue."""
    out: list[list[Word]] = []
    current: list[Word] = []

    for clause in _split_at(sentence_words, CLAUSE_END_PUNCT):
        candidate = current + clause
        if _fits_one_cue(candidate, rules):
            current = candidate
            continue

        if current:
            out.append(current)
            current = []

        if _fits_one_cue(clause, rules):
            current = clause
        else:
            out.extend(_greedy_split(clause, rules))

    if current:
        out.append(current)

    return out


def _split_at(words: list[Word], punct: tuple[str, ...]) -> list[list[Word]]:
    parts: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        if w.text.rstrip().endswith(punct):
            parts.append(current)
            current = []
    if current:
        parts.append(current)
    return parts


def _fits_one_cue(words: list[Word], rules: ChunkRules) -> bool:
    if not words:
        return True
    duration = words[-1].end - words[0].start
    if duration > rules.max_duration_seconds:
        return False
    return _layout_lines(words, rules) is not None


def _greedy_split(words: list[Word], rules: ChunkRules) -> list[list[Word]]:
    if not words:
        return []

    groups: list[list[Word]] = [[words[0]]]

    for prev, current_word in zip(words, words[1:]):
        candidate = groups[-1] + [current_word]
        gap = current_word.start - prev.end
        duration = candidate[-1].end - candidate[0].start

        breaks_cue = (
            gap > rules.gap_threshold_seconds
            or duration > rules.max_duration_seconds
            or _layout_lines(candidate, rules) is None
        )

        if breaks_cue:
            groups.append([current_word])
        else:
            groups[-1].append(current_word)

    return groups


def _layout_lines(
    words: list[Word],
    rules: ChunkRules,
    *,
    allow_overflow: bool = False,
) -> list[str] | None:
    """Greedy line wrap. Returns lines or None if exceeds max_lines_per_cue."""
    limit = rules.max_chars_per_line + (rules.max_chars_overflow if allow_overflow else 0)
    lines: list[str] = [""]

    for w in words:
        line = lines[-1]
        if not line:
            lines[-1] = w.text
        elif len(line) + 1 + len(w.text) <= limit:
            lines[-1] = f"{line} {w.text}"
        else:
            if len(lines) >= rules.max_lines_per_cue:
                return None
            lines.append(w.text)

    return lines


def _format_text(words: list[Word], rules: ChunkRules) -> str:
    lines = _layout_lines(words, rules, allow_overflow=True)
    if lines is None:
        return " ".join(w.text for w in words)
    return "\n".join(lines)


def _enforce_min_duration(cues: list[Cue], rules: ChunkRules) -> list[Cue]:
    if not cues:
        return cues

    adjusted: list[Cue] = []
    for i, cue in enumerate(cues):
        if cue.end - cue.start >= rules.min_duration_seconds:
            adjusted.append(cue)
            continue

        target_end = cue.start + rules.min_duration_seconds
        if i + 1 < len(cues):
            target_end = min(target_end, cues[i + 1].start - OVERLAP_SAFETY_GAP)
        target_end = max(target_end, cue.end)

        adjusted.append(Cue(index=cue.index, start=cue.start, end=target_end, text=cue.text))

    return adjusted
