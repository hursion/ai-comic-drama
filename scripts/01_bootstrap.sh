#!/usr/bin/env bash
# G1 bootstrap: dirs + venv + ComfyUI + minimal custom nodes (ROCm-safe)
set -euo pipefail

WORK="${WORK:-/mnt/workspace/ai-comic-drama}"
export WORK
COMFY_TAG="${COMFY_TAG:-v0.18.2}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "== AI Comic Drama bootstrap =="
echo "WORK=$WORK"
echo "COMFY_TAG=$COMFY_TAG"

mkdir -p "$WORK"/{logs,cache,scripts,pipeline,workflows,notebooks,configs,assets/{characters,scenes,voices},episodes/ep01/{keyframes,clips,audio,final,logs},models/comfy/{checkpoints,diffusion_models,unet,text_encoders,clip_vision,style_models,vae,loras,upscale_models,audio_encoders}}

# ---------------------------------------------------------------------------
# 1. Python venv (system-site-packages so the image's ROCm torch is visible)
# ---------------------------------------------------------------------------
if [ ! -d "$WORK/venv" ]; then
  echo "[1/5] creating venv (system-site-packages)"
  python -m venv --system-site-packages "$WORK/venv"
else
  echo "[1/5] venv exists"
fi
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
python -m pip install -U pip setuptools wheel -i "$PIP_INDEX"

# On a bare CUDA box there may be no system torch. On ModelScope AMD there is.
if ! python -c "import torch" >/dev/null 2>&1; then
  if [ "${INSTALL_TORCH:-0}" = "1" ]; then
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    echo "[1.5/5] installing torch from $TORCH_INDEX_URL"
    python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
  else
    echo "ERROR: torch not found. Use an image with torch, or set INSTALL_TORCH=1 TORCH_INDEX_URL=..." >&2
    exit 1
  fi
fi

python - <<'PY'
import json, os, platform, sys
work = os.environ.get("WORK", "/mnt/workspace/ai-comic-drama")
info = {
    "python": platform.python_version(),
    "exe": sys.executable,
    "platform": platform.platform(),
}
try:
    import torch
    info["torch"] = torch.__version__
    info["hip"] = getattr(torch.version, "hip", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(p.total_memory / 1e9, 1)
except Exception as e:
    info["torch_error"] = repr(e)
os.makedirs(os.path.join(work, "logs"), exist_ok=True)
with open(os.path.join(work, "logs", "env.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print(json.dumps(info, ensure_ascii=False, indent=2))
if not info.get("cuda_available"):
    print("WARN: torch.cuda.is_available() is False. Stop and fix torch before continuing.")
PY

# Pin whatever torch is already installed; never let pip replace it.
{
  python - <<'PY'
import torch
print("torch==" + torch.__version__)
PY
  python - <<'PY' 2>/dev/null || true
import torchvision
print("torchvision==" + torchvision.__version__)
PY
  python - <<'PY' 2>/dev/null || true
import torchaudio
print("torchaudio==" + torchaudio.__version__)
PY
} > "$WORK/constraints.txt"
echo "[2/5] constraints:"; cat "$WORK/constraints.txt"

# ---------------------------------------------------------------------------
# 2. ComfyUI
# ---------------------------------------------------------------------------
clone_with_mirror() {
  local url="$1" dest="$2" branch="${3:-}"
  local args=(--depth 1)
  [ -n "$branch" ] && args+=(--branch "$branch")
  if [ -d "$dest/.git" ]; then
    echo "  exists: $dest"
    return 0
  fi
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
    if git clone "${args[@]}" "$u" "$dest" 2>/dev/null; then
      return 0
    fi
    rm -rf "$dest"
  done
  echo "ERROR: clone failed for $url (tried mirrors). Upload a release zip manually." >&2
  return 1
}

if [ ! -d "$WORK/ComfyUI/.git" ]; then
  echo "[3/5] cloning ComfyUI ($COMFY_TAG)"
  clone_with_mirror "https://github.com/comfyanonymous/ComfyUI.git" "$WORK/ComfyUI" "$COMFY_TAG" || {
    echo "retry with master..."
    clone_with_mirror "https://github.com/comfyanonymous/ComfyUI.git" "$WORK/ComfyUI" ""
  }
else
  echo "[3/5] ComfyUI exists"
fi
git -C "$WORK/ComfyUI" rev-parse HEAD > "$WORK/logs/comfyui_commit.txt" 2>/dev/null || true

# Install requirements without torch/torchvision/torchaudio.
grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "$WORK/ComfyUI/requirements.txt" > "$WORK/requirements.no-torch.txt" || true
PIP_CONSTRAINT="$WORK/constraints.txt" python -m pip install -r "$WORK/requirements.no-torch.txt" \
  -i "$PIP_INDEX" --upgrade-strategy only-if-needed

# ---------------------------------------------------------------------------
# 3. Minimal custom nodes
# ---------------------------------------------------------------------------
mkdir -p "$WORK/ComfyUI/custom_nodes"
clone_node() {
  local name="$1" url="$2"
  if [ -d "$WORK/ComfyUI/custom_nodes/$name/.git" ]; then
    echo "  node exists: $name"
    return 0
  fi
  echo "  node: $name"
  clone_with_mirror "$url" "$WORK/ComfyUI/custom_nodes/$name" "" || true
}

echo "[4/5] custom nodes"
clone_node "ComfyUI-Manager"           "https://github.com/ltdrdata/ComfyUI-Manager.git"
clone_node "ComfyUI-VideoHelperSuite"  "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
clone_node "ComfyUI-KJNodes"           "https://github.com/kijai/ComfyUI-KJNodes.git"
clone_node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git"
if [ "${INSTALL_GGUF:-0}" = "1" ]; then
  clone_node "ComfyUI-GGUF" "https://github.com/city96/ComfyUI-GGUF.git"
fi

# Install node requirements, again excluding torch lines.
for req in "$WORK"/ComfyUI/custom_nodes/*/requirements.txt; do
  [ -f "$req" ] || continue
  echo "  deps: $req"
  grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "$req" > "$WORK/requirements.node.tmp" || true
  PIP_CONSTRAINT="$WORK/constraints.txt" python -m pip install -r "$WORK/requirements.node.tmp" \
    -i "$PIP_INDEX" --upgrade-strategy only-if-needed || echo "  WARN: deps failed for $req (check later)"
done
rm -f "$WORK/requirements.node.tmp"

# Known extras that nodes commonly need; CPU builds are fine.
PIP_CONSTRAINT="$WORK/constraints.txt" python -m pip install \
  opencv-python-headless imageio-ffmpeg av soundfile scipy \
  modelscope huggingface_hub accelerate sentencepiece \
  -i "$PIP_INDEX" || true

# ---------------------------------------------------------------------------
# 4. extra_model_paths.yaml
# ---------------------------------------------------------------------------
if [ ! -f "$WORK/workflows/extra_model_paths.yaml" ]; then
  cat > "$WORK/workflows/extra_model_paths.yaml" <<YAML
comic:
  base_path: $WORK/models/comfy
  checkpoints: checkpoints
  diffusion_models: diffusion_models
  unet: unet
  text_encoders: text_encoders
  clip_vision: clip_vision
  style_models: style_models
  vae: vae
  loras: loras
  upscale_models: upscale_models
  audio_encoders: audio_encoders
YAML
fi

# ---------------------------------------------------------------------------
# 5. Verify torch was not replaced
# ---------------------------------------------------------------------------
echo "[5/5] verify torch"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip  :", getattr(torch.version, "hip", None))
print("cuda :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda is not available after bootstrap; fix before continuing")
PY

echo
echo "bootstrap done."
echo "next: python scripts/02_download_models.py --groups image,image-t2i,video-5b,tts,llm-small,asr"
echo "then: bash scripts/03_start_comfyui.sh"
