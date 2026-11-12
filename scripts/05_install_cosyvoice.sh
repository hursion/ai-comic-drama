#!/usr/bin/env bash
# Best-effort CosyVoice2 install on ROCm. Timebox: 45 minutes.
# It uses a separate venv so TTS dependencies cannot break ComfyUI.
set -uo pipefail

WORK="${WORK:-/mnt/workspace/ai-comic-drama}"
COSY="${COSY:-$WORK/CosyVoice}"
VENV="${VENV_TTS:-$WORK/venv-tts}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "== CosyVoice2 install =="
echo "repo: $COSY"
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

python -m pip install -U pip setuptools wheel -i "$PIP_INDEX"

# Keep the image's ROCm torch; never let pip replace it.
{
  python - <<'PY'
import torch
print("torch==" + torch.__version__)
PY
  python - <<'PY' 2>/dev/null || true
import torchaudio
print("torchaudio==" + torchaudio.__version__)
PY
} > "$WORK/constraints.tts.txt"

# Minimal inference deps. deepspeed/onnxruntime-gpu are training/CUDA extras.
PIP_CONSTRAINT="$WORK/constraints.tts.txt" python -m pip install \
  conformer hyperpyyaml inflect librosa matplotlib numpy omegaconf \
  openai-whisper protobuf pyarrow pyworld requests soundfile \
  tensorboard tensorboardX transformers onnxruntime \
  -i "$PIP_INDEX" --upgrade-strategy only-if-needed || true

# Text normalization. pynini has no wheel for every Python; this is the usual blocker.
python -m pip install "pynini==2.1.5" -i "$PIP_INDEX" || {
  echo "WARN: pynini failed. Try: conda install -c conda-forge pynini==2.1.5"
  echo "      then re-run this script; without it CosyVoice text frontend may fail."
}
python -m pip install WeTextProcessing -i "$PIP_INDEX" --no-deps || true

# Import the package from the repo without letting pip touch torch.
PIP_CONSTRAINT="$WORK/constraints.tts.txt" python -m pip install -e "$COSY" --no-deps || true

COSY="$COSY" python - <<'PY'
import os, sys
sys.path.insert(0, os.environ["COSY"])
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
echo "  PYTHONPATH=$COSY python -m pipeline.tts_cosyvoice --text '测试' --prompt-text '参考台词' --prompt-wav assets/voices/lin.wav --out episodes/ep01/audio/test.wav"
echo "  # or, without a reference voice:"
echo "  PYTHONPATH=$COSY python -m pipeline.tts_cosyvoice --episode episodes/ep01/episode.json --sft-speaker 中文女"
