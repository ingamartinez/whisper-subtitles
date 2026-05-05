#!/usr/bin/env bash
# Bootstrap whisper-subtitles dependencies: ffmpeg, whisper.cpp, and models.
# Supports macOS (Homebrew) and Linux (apt + source build, CUDA auto-detected).
# Windows users: see README for the PowerShell path.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
VENDOR_DIR="$PROJECT_ROOT/vendor"
HF_BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
MODELS=(ggml-large-v3-turbo.bin ggml-large-v3.bin)

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*"; exit 1; }

install_mac() {
  command -v brew >/dev/null 2>&1 || fail "Homebrew is required on macOS. Install from https://brew.sh"

  log "Installing ffmpeg + whisper-cpp via Homebrew..."
  brew install ffmpeg whisper-cpp

  command -v whisper-cli >/dev/null 2>&1 \
    || warn "whisper-cli not on PATH yet — open a new shell or run: export PATH=\"\$(brew --prefix)/bin:\$PATH\""
}

install_linux() {
  log "Installing ffmpeg + build tools via apt..."
  sudo apt-get update
  sudo apt-get install -y ffmpeg git cmake build-essential

  mkdir -p "$VENDOR_DIR"
  local whisper_dir="$VENDOR_DIR/whisper.cpp"

  if [[ ! -d "$whisper_dir" ]]; then
    log "Cloning whisper.cpp into $whisper_dir..."
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp "$whisper_dir"
  fi

  local cmake_args=()
  if command -v nvcc >/dev/null 2>&1; then
    log "CUDA toolkit detected (nvcc) — building with CUDA support."
    cmake_args+=(-DGGML_CUDA=1)
  elif command -v nvidia-smi >/dev/null 2>&1; then
    warn "NVIDIA GPU detected but CUDA toolkit (nvcc) is missing — building CPU-only."
    warn "To enable GPU acceleration, install CUDA toolkit and re-run this script:"
    warn "  https://developer.nvidia.com/cuda-downloads (pick WSL-Ubuntu if on WSL2)"
  else
    log "No NVIDIA GPU detected — building CPU-only."
  fi

  log "Building whisper.cpp..."
  cmake -S "$whisper_dir" -B "$whisper_dir/build" "${cmake_args[@]}"
  cmake --build "$whisper_dir/build" -j --config Release

  mkdir -p "$HOME/.local/bin"
  ln -sf "$whisper_dir/build/bin/whisper-cli" "$HOME/.local/bin/whisper-cli"
  log "Linked whisper-cli into ~/.local/bin (open a new shell or 'source ~/.profile' to refresh PATH)."
}

download_models() {
  mkdir -p "$MODELS_DIR"
  for model in "${MODELS[@]}"; do
    local target="$MODELS_DIR/$model"
    if [[ -f "$target" ]]; then
      log "$model already present — skipping."
      continue
    fi
    log "Downloading $model..."
    curl -L --fail -o "$target.tmp" "$HF_BASE_URL/$model"
    mv "$target.tmp" "$target"
  done
}

main() {
  case "$(uname -s)" in
    Darwin) install_mac ;;
    Linux)  install_linux ;;
    *)      fail "Unsupported OS: $(uname -s). On Windows native, see README." ;;
  esac

  download_models
  log "Done. Run 'uv sync' next, then 'uv run whisper-subtitles --help'."
}

main "$@"
