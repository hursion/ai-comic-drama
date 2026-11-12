#!/usr/bin/env bash
# Pre-stop checks. This does NOT stop the cloud instance; click "关闭实例" yourself.
set -uo pipefail

WORK="${WORK:-/mnt/workspace/ai-comic-drama}"
PORT="${PORT:-8188}"

echo "== stop check =="
if [ -f "$WORK/logs/comfyui.pid" ]; then
  PID="$(cat "$WORK/logs/comfyui.pid" 2>/dev/null || true)"
  if [ -n "${PID}" ] && kill -0 "$PID" 2>/dev/null; then
    echo "stopping ComfyUI pid=$PID"
    kill "$PID" 2>/dev/null || true
    sleep 2
  else
    echo "ComfyUI not running"
  fi
else
  echo "no pid file"
fi

if command -v ss >/dev/null 2>&1 && ss -lntp 2>/dev/null | grep -q ":${PORT} "; then
  echo "WARN: port ${PORT} still listening:"
  ss -lntp 2>/dev/null | grep ":${PORT} " || true
fi

echo
echo "== disk =="
df -h /mnt/workspace 2>/dev/null || df -h
echo
echo "== outputs =="
find "$WORK/episodes" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' 2>/dev/null | sort | tail -20 || true
echo
echo "== last logs =="
tail -n 20 "$WORK/logs/comfyui.log" 2>/dev/null || true

sync || true
echo
echo "REMINDER:"
echo "  1) record the remaining quota on the ModelScope notebook page"
echo "  2) click '关闭实例' — do not rely on the 1h idle timer"
echo "  3) write today's Gate results into docs/CHECKLIST.md"
