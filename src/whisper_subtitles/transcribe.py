"""Transcription layer: audio -> word-level timestamps via WhisperX.

Pipeline:
  1. faster-whisper (CTranslate2) for transcription.
  2. wav2vec2 forced alignment for phoneme-level word boundaries.

Step 2 is what gives us frame-accurate timestamps for editing workflows;
cross-attention timestamps from Whisper alone drift ±100-300ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import whisperx


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
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    batch_size: int = 16,
) -> Transcription:
    """Transcribe an audio file and return word-level timestamps.

    `model_name` is a faster-whisper model identifier (e.g. 'large-v3',
    'large-v3-turbo'). Models are auto-downloaded to the HuggingFace cache
    on first use.

    `language` is an ISO code like 'es' or 'en'. None means auto-detect.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    audio = whisperx.load_audio(str(audio_path))

    asr_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
    )
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
    for segment in aligned.get("segments", []):
        for w in segment.get("words", []):
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if not text or start is None or end is None:
                continue
            words.append(Word(text=text, start=float(start), end=float(end)))

    return Transcription(language=detected_language, words=words)
