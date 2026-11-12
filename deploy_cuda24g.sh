#!/usr/bin/env bash
# 一键部署 cuda24g 环境（在魔搭实例终端粘贴执行）
# 用法: bash deploy_cuda24g.sh
set -euo pipefail

WORK=/mnt/workspace/ai-comic-drama
cd "$WORK"
echo "== 1. 修复 constraints =="
source venv/bin/activate
python -c "
import torch
with open('constraints.txt','w') as f:
    f.write('torch=='+torch.__version__.split('+')[0]+'\n')
    try:
        import torchvision; f.write('torchvision=='+torchvision.__version__.split('+')[0]+'\n')
    except: pass
    try:
        import torchaudio; f.write('torchaudio=='+torchaudio.__version__.split('+')[0]+'\n')
    except: pass
"
cat constraints.txt

echo "== 2. 修复 requirements（锁定 torchsde>=0.2.6）=="
sed -i 's/^torchsde$/torchsde>=0.2.6/' requirements.no-torch.txt 2>/dev/null || true

echo "== 3. 安装 ComfyUI 依赖 =="
PIP_CONSTRAINT=constraints.txt pip install -r requirements.no-torch.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade-strategy only-if-needed

echo "== 4. 补装可能遗漏的包 =="
pip install alembic sqlalchemy comfy_aimdo comfy-kitchen simpleeval kornia spandrel \
  PyOpenGL glfw opencv-python-headless imageio-ffmpeg soundfile \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade-strategy only-if-needed || true

echo "== 5. 安装自定义节点依赖 =="
for req in "$WORK"/ComfyUI/custom_nodes/*/requirements.txt; do
  [ -f "$req" ] || continue
  grep -viE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "$req" > /tmp/node_req.txt || true
  PIP_CONSTRAINT=constraints.txt pip install -r /tmp/node_req.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade-strategy only-if-needed || true
done

echo "== 6. 验证 torch =="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); assert torch.cuda.is_available()"

echo "== 7. 下载模型 (cuda24g ~64GB) =="
export WORK MODEL_PROFILE=cuda24g
export MIN_FREE_GB=75 INSTALL_GGUF=1
export COMFY_VRAM_MODE=lowvram COMFY_RESERVE_VRAM=1.5
export COMFY_EXTRA_ARGS="--cache-none"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_STEPS=20 IMAGE_CFG=4.0
export VIDEO_WIDTH=480 VIDEO_HEIGHT=832 VIDEO_FRAMES=81 VIDEO_FPS=16
python scripts/02_download_models.py --profile cuda24g --yes

echo "== 8. 启动 ComfyUI =="
bash scripts/03_start_comfyui.sh

echo "== 部署完成 =="
echo "ComfyUI: http://127.0.0.1:8188"
echo "下一步: 打开 notebooks/01_episode.ipynb 开始生成"