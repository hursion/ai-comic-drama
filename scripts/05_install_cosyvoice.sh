#!/usr/bin/env bash
# Best-effort CosyVoice2 install on ROCm/CUDA. Timebox: 45 minutes.
# It uses a separate venv so TTS dependencies cannot break ComfyUI.
set -uo pipefail

WORK="${WORK:-/mnt/workspace/ai-comic-drama}"
COSY="${COSY:-$WORK/CosyVoice}"
MATCHA="${MATCHA:-$WORK/third_party/Matcha-TTS}"
VENV="${VENV_TTS:-$WORK/venv-tts}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "== CosyVoice2 install =="
echo "repo: $COSY"
echo "matcha: $MATCHA"
echo "venv: $VENV"

clone_with_mirror() {
  local url="$1" dest="$2"
  [ -d "$dest/.git" ] && return 0
  for prefix in "" "https://ghfast.top/" "https://gh-proxy.com/" "https://gitclone.com/github.com/"; do
    local u="$url"
    if [ -n "$prefix" ]; then
      if [[ "$prefix" == *"gitclone.com"* ]]; then
        u="$prefix/${url#https://github.com/}"
      else
        u="$prefix$url"
      fi
    fi
    echo "  try: $u"
    if git clone --recursive "${u}" "$dest"; then
      return 0
    fi
    rm -rf "$dest"
  done
  return 1
}

if [ ! -d "$COSY/.git" ]; then
  clone_with_mirror "https://github.com/FunAudioLLM/CosyVoice.git" "$COSY" || {
    echo "ERROR: clone failed. Download the repo zip manually and retry." >&2
    exit 1
  }
fi
# CosyVoice imports matcha at runtime; it is PYTHONPATH-only (no pip install).
if [ ! -d "$MATCHA/.git" ]; then
  mkdir -p "$(dirname "$MATCHA")"
  clone_with_mirror "https://github.com/shivammehta25/Matcha-TTS.git" "$MATCHA" || {
    echo "WARN: Matcha-TTS clone failed; CosyVoice2 import will fail without it." >&2
  }
fi

BASE_PYTHON="${BASE_PYTHON:-$WORK/venv/bin/python}"
if [ ! -x "$BASE_PYTHON" ]; then
  BASE_PYTHON="$(command -v python)"
fi
if [ ! -d "$VENV" ]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# If torch lives in the main project venv (bare CUDA box), expose it to venv-tts.
MAIN_SITE="$("$BASE_PYTHON" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
VENV_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
if [ -n "$MAIN_SITE" ] && [ "$MAIN_SITE" != "$VENV_SITE" ]; then
  echo "$MAIN_SITE" > "$VENV_SITE/main_venv.pth"
fi

# setuptools<81: pkg_resources (imported by cosyvoice.dataset.processor) was
# removed in setuptools 81+, so a newer setuptools breaks CosyVoice2 init.
python -m pip install -U pip "setuptools<81" wheel -i "$PIP_INDEX"

# Keep the image's torch; never let pip replace it.
# Strip the local version label (+cu130/rocm...): mirrors only carry the bare
# version, and a pinned +label makes every torch-dependent resolve impossible.
{
  python - <<'PY'
import torch
print("torch==" + torch.__version__.split("+")[0])
PY
  python - <<'PY' 2>/dev/null || true
import torchaudio
print("torchaudio==" + torchaudio.__version__.split("+")[0])
PY
} > "$WORK/constraints.tts.txt"

# Minimal inference deps. deepspeed/gradio/onnxruntime-gpu are training/CUDA extras.
# numpy<2.5 keeps numba (openai-whisper) importable; lightning+diffusers are
# imported by cosyvoice.flow at runtime.
PIP_CONSTRAINT="$WORK/constraints.tts.txt" python -m pip install \
  conformer hyperpyyaml inflect librosa matplotlib "numpy<2.5" omegaconf \
  openai-whisper protobuf pyarrow pyworld requests soundfile \
  tensorboard tensorboardX transformers onnxruntime lightning diffusers \
  -i "$PIP_INDEX" --upgrade-strategy only-if-needed || true

# Text normalization is OPTIONAL. CosyVoice's frontend tries ttsfrd, then
# `from wetext import Normalizer`, and falls back to no normalization when
# neither imports — TTS still works (fine for clean dialogue). The wetext wheel
# is on PyPI but not every mirror carries it, and it may need pynini/OpenFst to
# build, so bound the attempt and never block on it.
python -m pip install wetext -i "$PIP_INDEX" --no-deps --timeout 30 --retries 1 || \
  python -m pip install wetext -i https://pypi.org/simple --no-deps --timeout 30 --retries 1 || \
  echo "NOTE: wetext unavailable; CosyVoice runs without text normalization."

# CosyVoice is not a pip project (no setup.py/pyproject.toml); import it via
# PYTHONPATH together with Matcha-TTS. Never `pip install -e` the repo.
COSY="$COSY" MATCHA="$MATCHA" python - <<'PY'
import os, sys
sys.path.insert(0, os.environ["COSY"])
sys.path.insert(0, os.environ["MATCHA"])
try:
    import torch, torchaudio
    print("torch:", torch.__version__, "hip:", torch.version.hip, "cuda:", torch.cuda.is_available())
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print("CosyVoice2 import: OK")
except Exception as e:
    print("CosyVoice2 import FAILED:", repr(e))
    raise SystemExit(1)
PY

echo
echo "next:"
echo "  source venv-tts/bin/activate"
echo "  PYTHONPATH=$COSY:$MATCHA python -m pipeline.tts_cosyvoice --text '测试' --prompt-text '参考台词' --prompt-wav assets/voices/lin.wav --out episodes/ep01/audio/test.wav"
echo "  # or, without a reference voice:"
echo "  PYTHONPATH=$COSY:$MATCHA python -m pipeline.tts_cosyvoice --episode episodes/ep01/episode.json --sft-speaker 中文女"