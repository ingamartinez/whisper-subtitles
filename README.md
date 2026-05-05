# whisper-subtitles

Generate `.srt` subtitles from videos using local Whisper (`whisper.cpp`).
No API calls, no data leaving your machine.

## Pipeline

```
video → ffmpeg (WAV 16kHz mono) → whisper.cpp (JSON, word-level) → chunker → .srt
```

The intermediate JSON with per-word timestamps is the source of truth.
Chunking rules turn that into properly formatted SRT cues — and that
seam is where per-client rules will plug in later (Phase 2).

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv)
- `ffmpeg`
- `whisper.cpp` compiled locally (Metal on Mac, CUDA on Windows/Linux)

## Setup

```bash
./scripts/setup.sh
uv sync
```

The setup script installs `ffmpeg`, builds `whisper.cpp` with the right
backend for your platform, and downloads both models into `models/`:

- `ggml-large-v3-turbo.bin` (~1.5GB) — multilingual, fast (default)
- `ggml-large-v3.bin` (~3GB) — multilingual, max quality

## Usage

```bash
# Default: turbo model, auto-detect language
uv run whisper-subtitles transcribe path/to/video.mp4

# Pin model + language
uv run whisper-subtitles transcribe video.mp4 --model large-v3 --language es

# Custom output path
uv run whisper-subtitles transcribe video.mp4 -o subs/video.srt
```

## Project layout

```
src/whisper_subtitles/
  cli.py          # typer entry point
  audio.py        # ffmpeg → WAV
  transcribe.py   # whisper.cpp → words
  chunker.py      # words → cues (rules live here)
  srt.py          # cues → .srt
```

## Roadmap

- **Phase 1** — core pipeline with default chunking rules.
- **Phase 2** — dashboard with per-client rule profiles (glossaries,
  text replacements, line/duration limits, etc).
