#!/usr/bin/env bash
# G0 preflight: GPU / ROCm / torch / disk / network / tools
set -uo pipefail

WORK="${WORK:-/mnt/workspace/ai-comic-drama}"
MIN_FREE_GB="${MIN_FREE_GB:-95}"
mkdir -p "$WORK/logs" 2>/dev/null || true
echo "============================================================"
echo " AI Comic Drama · preflight"
echo " time: $(date '+%F %T %z')"
echo " work: $WORK"
echo "============================================================"

section() { printf "\n--- %s ---\n" "$1"; }

section "OS / CPU / RAM"
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "os: ${PRETTY_NAME:-unknown}"
else
  uname -a
fi
echo "kernel: $(uname -r)"
echo "cpu: $(nproc) vCPU"
free -h 2>/dev/null || true

section "DISK"
df -h / /mnt/workspace 2>/dev/null || df -h
DISK_TARGET="${DISK_TARGET:-$(dirname "$WORK")}"
if AVAIL_GB=$(df -Pk "$DISK_TARGET" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}'); then
  if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt "$MIN_FREE_GB" ]; then
    echo "WARN: $DISK_TARGET has ${AVAIL_GB}GB free; profile needs >= ${MIN_FREE_GB}GB. Clean up first."
  fi
fi

section "NVIDIA / CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || nvidia-smi | head -20
else
  echo "nvidia-smi: not found (AMD/ROCm profile?)"
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -1
else
  echo "nvcc: not found (toolkit optional; PyTorch wheels bundle CUDA runtime)"
fi

section "ROCm"
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo 2>/dev/null | grep -E "Marketing Name|gfx" | head -20
else
  echo "rocminfo: MISSING"
fi
if [ -f /opt/rocm/.info/version ]; then
  echo "rocm version: $(cat /opt/rocm/.info/version)"
elif [ -f /opt/rocm/.info/version-dev ]; then
  echo "rocm version: $(cat /opt/rocm/.info/version-dev)"
else
  echo "rocm version: file not found (check /opt/rocm)"
fi
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi 2>/dev/null | head -40
else
  echo "rocm-smi: MISSING"
fi

section "DEVICE NODES"
ls -l /dev/kfd /dev/dri 2>/dev/null || echo "WARN: /dev/kfd or /dev/dri not found"
id

section "PYTHON / TORCH"
python - <<'PY'
import platform
print("python:", platform.python_version())
print("exe   :", __import__("sys").executable)
try:
    import torch
    print("torch :", torch.__version__)
    print("hip   :", getattr(torch.version, "hip", None))
    print("cuda  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device:", torch.cuda.get_device_name(0))
        print("vram  : %.1f GB" % (p.total_memory / 1e9))
        print("gcn   :", getattr(p, "gcnArchName", "n/a"))
except Exception as e:
    print("torch check FAILED:", repr(e))
PY

section "TOOLS"
for c in git git-lfs ffmpeg ffprobe curl wget unzip; do
  if command -v "$c" >/dev/null 2>&1; then
    printf "%-8s %s\n" "$c" "$(command -v "$c")"
  else
    printf "%-8s MISSING\n" "$c"
  fi
done

section "FONTS"
if command -v fc-list >/dev/null 2>&1; then
  if fc-list :lang=zh 2>/dev/null | head -1 | grep -q .; then
    echo "CJK font: OK"
  else
    echo "WARN: no CJK font; subtitle burn-in will fail. Install: apt-get install -y fonts-noto-cjk"
  fi
else
  echo "WARN: fc-list missing; install fontconfig + fonts-noto-cjk for Chinese subtitles"
fi

section "SUDO"
if command -v sudo >/dev/null 2>&1; then
  if sudo -n true >/dev/null 2>&1; then
    echo "sudo: passwordless OK"
  else
    echo "sudo: present but requires password (may still work interactively)"
  fi
else
  echo "sudo: MISSING (check if running as root)"
fi

section "NETWORK"
if command -v curl >/dev/null 2>&1; then
  curl -sS -o /dev/null -w "modelscope: %{http_code}  %{time_total}s\n" --max-time 15 https://www.modelscope.cn || true
  curl -sS -o /dev/null -w "github   : %{http_code}  %{time_total}s\n" --max-time 15 https://github.com || true
  curl -sS -o /dev/null -w "pypi     : %{http_code}  %{time_total}s\n" --max-time 15 https://pypi.tuna.tsinghua.edu.cn/simple/ || true
fi

section "SUMMARY"
echo "把本输出保存到 logs/preflight_$(date +%Y%m%d_%H%M).log"
echo "G0 通过标准：torch cuda True / device 符合 profile / 磁盘可用 >= ${MIN_FREE_GB}GB"
echo "MI300X: vram ~192GB；cuda24g: vram ~24GB + RAM >= 32GB"
