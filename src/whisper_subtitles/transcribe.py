"""Transcription layer: audio -> word-level timestamps via WhisperX.

Pipeline:
  1. faster-whisper (CTranslate2) for transcription.
  2. wav2vec2 forced alignment for phoneme-level word boundaries.
  3. Triple verification before accepting a word:
     - alignment score (wav2vec2 found the phonemes acoustically)
     - duration sanity (no stretched alignments over silence gaps)
     - audio RMS energy (the audio actually has signal in that window)

Step 2 gives us frame-accurate timestamps. Step 3 catches Whisper's most
common failure mode: hallucinating filler words during silence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import whisperx

SAMPLE_RATE = 16000  # whisperx.load_audio resamples to 16kHz


def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation/whitespace for dedupe comparison."""
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


def _dedupe_overlapping(words: list["Word"]) -> tuple[list["Word"], int]:
    """Drop words with the same normalized text whose time ranges overlap.

    Repetitions ("no, no") have consecutive non-overlapping ranges and are
    preserved. Duplicates from diarization overlap heavily and are removed
    (keeping the higher-score one).
    """
    kept: list[Word] = []
    dropped = 0
    for w in words:
        norm = _normalize_text(w.text)
        if not norm:
            kept.append(w)
            continue
        is_dup = False
        for i, prev in enumerate(kept):
            if _normalize_text(prev.text) != norm:
                continue
            ovl = min(w.end, prev.end) - max(w.start, prev.start)
            if ovl <= 0:
                continue
            shorter = min(w.end - w.start, prev.end - prev.start)
            if shorter <= 0:
                continue
            if ovl / shorter < 0.3:
                continue
            is_dup = True
            if w.score > prev.score:
                kept[i] = w
            break
        if not is_dup:
            kept.append(w)
        else:
            dropped += 1
    kept.sort(key=lambda x: x.start)
    return kept, dropped


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float
    score: float


@dataclass(frozen=True, slots=True)
class Transcription:
    language: str
    words: list[Word]


def _rms(audio: np.ndarray, start: float, end: float) -> float:
    """RMS energy of `audio` (16kHz mono float32) over the [start, end] window."""
    i0 = max(0, int(start * SAMPLE_RATE))
    i1 = min(len(audio), int(end * SAMPLE_RATE))
    if i1 <= i0:
        return 0.0
    chunk = audio[i0:i1]
    return float(np.sqrt(np.mean(chunk * chunk)))


def transcribe(
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    batch_size: int = 16,
    vad_onset: float = 0.200,
    vad_offset: float = 0.200,
    no_speech_threshold: float = 0.3,
    min_alignment_score: float = 0.05,
    max_word_duration_s: float = 2.0,
    rms_silence_threshold: float = 0.025,
    rms_check_score_below: float = 0.50,
    hf_token: str | None = None,
    diarize: bool = False,
    turn_pad_ms: int = 100,
    min_turn_duration_ms: int = 400,
) -> Transcription:
    """Transcribe an audio file and return word-level timestamps.

    Filters applied (any failure drops the word):
      - alignment score < min_alignment_score
      - duration > max_word_duration_s
      - score < rms_check_score_below AND RMS(window) < rms_silence_threshold
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    audio = whisperx.load_audio(str(audio_path))

    asr_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
        asr_options={"no_speech_threshold": no_speech_threshold},
    )

    if diarize and hf_token:
        from whisperx.diarize import DiarizationPipeline

        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        diarize_df = diarize_model(audio)
        print(f"[transcribe] diarization: {len(diarize_df)} turns", file=sys.stderr)

        pad = turn_pad_ms / 1000.0
        min_turn = min_turn_duration_ms / 1000.0
        all_segments: list[dict] = []
        for _, turn in diarize_df.iterrows():
            t_start = float(turn["start"])
            t_end = float(turn["end"])
            speaker = str(turn.get("speaker", "?"))
            if t_end - t_start < min_turn:
                continue
            chunk_start = max(0.0, t_start - pad)
            chunk_end = min(len(audio) / SAMPLE_RATE, t_end + pad)
            i0 = int(chunk_start * SAMPLE_RATE)
            i1 = int(chunk_end * SAMPLE_RATE)
            chunk = audio[i0:i1]
            if len(chunk) < SAMPLE_RATE * 0.3:
                continue
            sub = asr_model.transcribe(chunk, batch_size=batch_size)
            for seg in sub.get("segments", []):
                seg = dict(seg)
                seg["start"] = float(seg["start"]) + chunk_start
                seg["end"] = float(seg["end"]) + chunk_start
                seg["speaker"] = speaker
                all_segments.append(seg)

        all_segments.sort(key=lambda s: s["start"])
        asr_result = {"segments": all_segments, "language": language or "es"}
        detected_language = asr_result["language"]
        print(f"[transcribe] per-speaker yielded {len(all_segments)} segments", file=sys.stderr)
    else:
        asr_result = asr_model.transcribe(audio, batch_size=batch_size)
        detected_language = asr_result.get("language", language or "unknown")

    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    aligned = whisperx.align(
        asr_result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    words: list[Word] = []
    drops_score: list[str] = []
    drops_duration: list[str] = []
    drops_rms: list[str] = []
    for segment in aligned.get("segments", []):
        for w in segment.get("words", []):
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            score = w.get("score")
            if not text or start is None or end is None:
                continue
            start_f = float(start)
            end_f = float(end)
            score_f = float(score) if score is not None else 0.0
            duration = end_f - start_f
            tag = f"{start_f:7.3f}-{end_f:7.3f} dur={duration:.2f}s score={score_f:.3f} {text!r}"

            if score_f < min_alignment_score:
                drops_score.append(tag)
                continue
            if duration > max_word_duration_s:
                drops_duration.append(tag)
                continue
            if score_f < rms_check_score_below:
                rms = _rms(audio, start_f, end_f)
                if rms < rms_silence_threshold:
                    drops_rms.append(f"{tag} rms={rms:.4f}")
                    continue

            words.append(Word(text=text, start=start_f, end=end_f, score=score_f))

    for label, drops in (("score", drops_score), ("duration", drops_duration), ("rms", drops_rms)):
        if drops:
            print(f"[transcribe] dropped {len(drops)} by {label}:", file=sys.stderr)
            for d in drops:
                print(f"  {d}", file=sys.stderr)

    if diarize:
        words, dup_dropped = _dedupe_overlapping(words)
        if dup_dropped:
            print(f"[transcribe] deduped {dup_dropped} overlapping duplicates from diarization", file=sys.stderr)

    return Transcription(language=detected_language, words=words)
