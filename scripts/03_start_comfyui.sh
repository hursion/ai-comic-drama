#!/usr/bin/env bash
# Start ComfyUI with ROCm-friendly defaults.
set -euo pipefail

WORK="${WORK:-/mnt/workspace/ai-comic-drama}"
PORT="${PORT:-8188}"
LOG="$WORK/logs/comfyui.log"

# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
cd "$WORK/ComfyUI"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$WORK/cache/miopen}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK/cache/triton}"
export HF_HOME="${HF_HOME:-$WORK/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$WORK/cache/modelscope}"
mkdir -p "$MIOPEN_USER_DB_PATH" "$TRITON_CACHE_DIR" "$HF_HOME" "$MODELSCOPE_CACHE" "$WORK/logs"

# Detect vendor/VRAM so the same script works on MI300X and 24GB CUDA cards.
GPU_INFO="$(python - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        vendor = "hip" if getattr(torch.version, "hip", None) else "cuda"
        print(vendor, round(p.total_memory / 1e9, 1))
    else:
        print("none 0")
except Exception:
    print("none 0")
PY
)"
VENDOR="$(printf '%s' "$GPU_INFO" | awk '{print $1}')"
VRAM_GB="$(printf '%s' "$GPU_INFO" | awk '{print $2}')"
VRAM_INT="${VRAM_GB%%.*}"
if [ "$VENDOR" = "hip" ]; then
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
else
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

if [ -z "${COMFY_VRAM_MODE:-}" ]; then
  if [ "${VRAM_INT:-0}" -ge 80 ]; then
    COMFY_VRAM_MODE=highvram
  elif [ "${VRAM_INT:-0}" -ge 48 ]; then
    COMFY_VRAM_MODE=normalvram
  else
    COMFY_VRAM_MODE=lowvram
  fi
fi
VRAM_ARGS=()
case "$COMFY_VRAM_MODE" in
  highvram) VRAM_ARGS+=(--highvram) ;;
  normalvram) VRAM_ARGS+=(--normalvram) ;;
  novram) VRAM_ARGS+=(--novram) ;;
  *) VRAM_ARGS+=(--lowvram) ;;
esac
COMFY_RESERVE_VRAM="${COMFY_RESERVE_VRAM:-0}"
if awk "BEGIN{exit !($COMFY_RESERVE_VRAM > 0)}"; then
  VRAM_ARGS+=(--reserve-vram "$COMFY_RESERVE_VRAM")
fi
read -r -a EXTRA_ARGS <<< "${COMFY_EXTRA_ARGS:-}"
echo "gpu: vendor=$VENDOR vram=${VRAM_GB}GB mode=$COMFY_VRAM_MODE reserve=${COMFY_RESERVE_VRAM}GB extra=${COMFY_EXTRA_ARGS:-none}"

if [ -f "$WORK/logs/comfyui.pid" ]; then
  OLD_PID="$(cat "$WORK/logs/comfyui.pid" 2>/dev/null || true)"
  if [ -n "${OLD_PID}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "stopping old ComfyUI pid=$OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 3
  fi
fi

if command -v ss >/dev/null 2>&1 && ss -lntp 2>/dev/null | grep -q ":${PORT} "; then
  echo "ERROR: port ${PORT} is already in use:"
  ss -lntp 2>/dev/null | grep ":${PORT} " || true
  exit 1
fi

echo "starting ComfyUI on 0.0.0.0:${PORT} ..."
nohup python main.py \
  --listen 0.0.0.0 \
  --port "${PORT}" \
  --use-pytorch-cross-attention \
  "${VRAM_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --preview-method none \
  --disable-auto-launch \
  --extra-model-paths-config "$WORK/workflows/extra_model_paths.yaml" \
  > "$LOG" 2>&1 &
echo $! > "$WORK/logs/comfyui.pid"

for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/system_stats" >/dev/null 2>&1; then
    echo "ComfyUI is up (pid=$(cat "$WORK/logs/comfyui.pid"))"
    curl -s "http://127.0.0.1:${PORT}/system_stats" | python -m json.tool || true
    exit 0
  fi
  sleep 2
done

echo "ERROR: ComfyUI did not become ready in 120s. Last log lines:"
tail -n 80 "$LOG" || true
exit 1
