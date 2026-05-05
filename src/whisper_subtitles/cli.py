"""CLI entry point. Usage: `whisper-subtitles transcribe <video>`."""

from __future__ import annotations

import os
import tempfile
from enum import Enum
from pathlib import Path

import typer

from whisper_subtitles.audio import extract_audio
from whisper_subtitles.chunker import chunk_words
from whisper_subtitles.presets import PRESETS
from whisper_subtitles.srt import write_srt
from whisper_subtitles.transcribe import transcribe as run_whisper

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Generate .srt subtitles from a video using local Whisper.",
)


@app.callback()
def _root() -> None:
    """Forces typer into multi-command mode so single subcommands stay subcommands."""


def _load_dotenv(path: Path) -> None:
    """Best-effort .env loader. Quietly does nothing if file missing or unreadable."""
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


class ModelChoice(str, Enum):
    turbo = "turbo"
    large_v3 = "large-v3"


class Preset(str, Enum):
    traditional = "traditional"
    social = "social"
    karaoke = "karaoke"
    dilan = "dilan"


MODEL_NAMES = {
    ModelChoice.turbo: "large-v3-turbo",
    ModelChoice.large_v3: "large-v3",
}

HF_TOKEN_VARS = ("HUGGINFACE_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN")


@app.command()
def transcribe(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Input video file."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output .srt path. Defaults to <video>.srt."),
    model: ModelChoice = typer.Option(ModelChoice.turbo, "--model", "-m", help="Whisper model to use."),
    language: str | None = typer.Option(None, "--language", "-l", help="ISO language code (es, en, ...). Omit for auto-detect."),
    preset: Preset = typer.Option(Preset.traditional, "--preset", "-p", help="Subtitle style."),
    diarize: bool = typer.Option(False, "--diarize/--no-diarize", help="Run speaker diarization and transcribe each speaker turn separately. Requires HF token in env."),
    normalize: bool = typer.Option(False, "--normalize/--no-normalize", help="Apply dynaudnorm to audio before transcription."),
) -> None:
    """Transcribe a video and write an .srt next to it."""
    _load_dotenv(Path.cwd() / ".env")

    hf_token: str | None = None
    if diarize:
        for var in HF_TOKEN_VARS:
            if os.environ.get(var):
                hf_token = os.environ[var]
                break
        if not hf_token:
            typer.echo(
                f"--diarize needs a HuggingFace token in one of: {', '.join(HF_TOKEN_VARS)} (in env or .env).",
                err=True,
            )
            raise typer.Exit(1)

    output_path = output or video.with_suffix(".srt")
    model_name = MODEL_NAMES[model]

    with tempfile.TemporaryDirectory(prefix="whisper-subtitles-") as tmp:
        typer.echo(f"Extracting audio from {video.name} (normalize={normalize})...")
        wav_path = extract_audio(video, Path(tmp), normalize=normalize)

        mode = "diarize+per-speaker" if diarize else "single-pass"
        typer.echo(f"Transcribing with model={model.value}, language={language or 'auto'}, mode={mode}...")
        transcription = run_whisper(wav_path, model_name, language, hf_token=hf_token, diarize=diarize)

    typer.echo(f"Detected language: {transcription.language} ({len(transcription.words)} words)")

    cues = chunk_words(transcription.words, PRESETS[preset.value])
    write_srt(cues, output_path)

    typer.echo(f"Wrote {len(cues)} cues to {output_path} (preset={preset.value})")


def main() -> None:
    app()
