# Apple M4 Pro 兼容手册（MPS 后端）

## 1. 可行性结论

| 环节 | 24GB 统一内存 | 48GB 统一内存 | 备注 |
|---|---|---|---|
| 图像生成 (Qwen-Image GGUF Q4) | 可行，~4 min/张 | 可行，~3 min/张 | 768×1344, 20-step |
| 视频生成 (Wan2.2 5B) | 勉强，~30 min/片段 | 可行，~20 min/片段 | 384×672, 49f；极慢 |
| 配音 (CosyVoice2) | CPU 模式可行 | CPU 模式可行 | ~实时速度 |
| 分镜 LLM (Qwen3-4B) | 可行 | 可行 | MPS 或 CPU |
| 合成 (ffmpeg) | 可行 | 可行 | CPU，秒级 |

**核心限制**：MPS 不支持 fp8 数据类型。所有 `fp8_e4m3fn` 权重必须替换为 fp16/bf16 或 GGUF。

**推荐策略**：图像用 AI 生成，视频用 **Ken Burns 动效**（ffmpeg zoompan）代替 AI 视频。
漫剧本身是"动态漫画"风格，镜头推拉摇移完全够用，且速度快 100 倍。

## 2. 硬件与内存预算

M4 Pro 12+16 核，统一内存架构（CPU/GPU 共享）：

| 配置 | 可用内存（扣除 macOS） | 图像峰值 | 视频峰值 |
|---|---|---|---|
| 24GB | ~18GB | GGUF Q4 UNet 12.8GB + 激活 ~2GB | 5B fp16 10GB + 激活 ~3GB |
| 48GB | ~42GB | Q4 UNet 12.8GB + bf16 TE 14GB（顺序加载） | 5B 10GB + fp16 TE 11.4GB（顺序） |

ComfyUI `--lowvram` 会顺序加载/卸载模型，峰值 = max(TE, UNet) + VAE + 激活。

### 磁盘预算（512GB SSD 充裕）

| 模型 | 大小 | 用途 |
|---|---|---|
| qwen-image-edit-2511-Q4_K_M.gguf | 12.8GB | 图像编辑 |
| qwen-image-Q4_K_M.gguf | 12.84GB | T2I（可选） |
| qwen_2.5_vl_7b GGUF Q4_K_M | ~4.5GB | 图像 TE（24GB 必选） |
| qwen_2.5_vl_7b bf16 safetensors | ~14GB | 图像 TE（48GB 可选） |
| wan2.2_ti2v_5B_fp16.safetensors | 10GB | 视频（可选） |
| umt5-xxl-encoder-Q8_0.gguf | ~5.7GB | 视频 TE（24GB） |
| umt5_xxl_fp16.safetensors | 11.4GB | 视频 TE（48GB） |
| qwen_image_vae.safetensors | 0.25GB | VAE |
| wan2.2_vae.safetensors | 1.41GB | 视频 VAE |
| CosyVoice2-0.5B | 2GB | TTS |
| Qwen3-4B | 8GB | 分镜 |
| **合计（24GB 精简）** | **~46GB** | |
| **合计（48GB 完整）** | **~67GB** | |

## 3. 环境安装（macOS）

### 3.1 前置

```bash
# Homebrew（如果没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 git ffmpeg

# 确认 Python
python3.12 --version
```

### 3.2 部署

```bash
# 解压项目到持久目录
mkdir -p ~/ai-comic-drama && cd ~/ai-comic-drama
# （上传 ai-comic-drama.zip 并解压）

source configs/m4pro.env
bash scripts/00_preflight.sh
bash scripts/01_bootstrap.sh
source venv/bin/activate
```

bootstrap 在 macOS 上的行为：
- 创建 venv（不需要 `--system-site-packages`，macOS 无预装 torch）
- `pip install torch torchvision torchaudio`（PyPI 默认含 MPS）
- 克隆 ComfyUI + ComfyUI-GGUF
- 安装依赖（无 CUDA 相关包）

### 3.3 下载模型

```bash
# 24GB 精简（不含视频，用 Ken Burns 代替）
python scripts/02_download_models.py --profile m4pro-min

# 48GB 完整（含视频）
python scripts/02_download_models.py --profile m4pro
```

### 3.4 启动 ComfyUI

```bash
bash scripts/03_start_comfyui.sh
# 自动检测 MPS → --force-fp16 --use-split-cross-attention --lowvram
curl -s http://127.0.0.1:8188/system_stats | python -m json.tool
```

## 4. 与 CUDA 方案的关键差异

| 项目 | CUDA 24GB | M4 Pro MPS |
|---|---|---|
| 后端 | CUDA + cuDNN | Metal Performance Shaders |
| fp8 权重 | 支持（原生） | **不支持，必须换 GGUF/fp16** |
| 图像 TE | fp8_scaled (9.4GB) | GGUF Q4 (~4.5GB) 或 bf16 (14GB) |
| 视频 TE | fp8 (6.7GB) | GGUF Q8 (~5.7GB) 或 fp16 (11.4GB) |
| 注意力 | pytorch-cross-attention | **split-cross-attention**（MPS 兼容） |
| 精度 | fp16/bf16 自动 | **--force-fp16**（MPS bf16 支持不完整） |
| 内存管理 | PYTORCH_CUDA_ALLOC_CONF | PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 |
| CPU 回退 | 无 | PYTORCH_ENABLE_MPS_FALLBACK=1（部分算子） |
| 图像速度 | ~30s/张 (A10) | **~4 min/张** |
| 视频速度 | ~2 min/片段 (A10) | **~20-30 min/片段** |
| 14B / S2V | 不推荐 | **不可行** |

## 5. 工作流导出要点

### 5.1 图像工作流（GGUF）

与 cuda24g 相同，但文本编码器节点不同：

- 24GB：使用 `CLIPLoader (GGUF)` 加载 `qwen_2.5_vl_7b-Q4_K_M.gguf`
  - 如果 ComfyUI-GGUF 无 CLIP GGUF 节点，改用 `UNETLoader` + 手动 TE 分离
  - 备选：用 bf16 TE + `--lowvram` 顺序加载（24GB 勉强可行）
- 48GB：使用标准 `CLIPLoader` 加载 `qwen_2.5_vl_7b.safetensors`（bf16）

节点命名契约不变：POSITIVE / NEGATIVE / REF1 / REF2 / SAVE

### 5.2 视频工作流（如果启用）

- Diffusion：标准 `UNETLoader` 加载 `wan2.2_ti2v_5B_fp16.safetensors`
- TE 24GB：`CLIPLoader (GGUF)` 加载 `umt5-xxl-encoder-Q8_0.gguf`
- TE 48GB：标准 `CLIPLoader` 加载 `umt5_xxl_fp16.safetensors`
- VAE：`wan2.2_vae.safetensors`
- 分辨率：384×672，49 帧，12fps（约 4 秒）
- **不要**用 fp8 的任何文件

### 5.3 Ken Burns 动效（推荐替代 AI 视频）

```bash
# 对每张关键帧生成 4 秒推拉镜头
ffmpeg -loop 1 -i keyframe_01.png -vf "zoompan=z='min(zoom+0.001,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=100:s=768x1344:fps=25" -t 4 -c:v libx264 -pix_fmt yuv420p clip_01.mp4
```

优点：秒级生成、无显存压力、风格统一。
缺点：无真实运动（人物不会动），适合漫剧/有声漫画风格。

## 6. 性能预估与时间预算

### 6.1 纯 AI 视频方案（48GB，慢）

| 阶段 | 数量 | 单件耗时 | 合计 |
|---|---|---|---|
| 关键帧 | 8 张 | 4 min | 32 min |
| 视频片段 | 8 个 | 25 min | 200 min |
| 配音 | 8 条 | 30s | 4 min |
| 合成 | 1 | 2 min | 2 min |
| **40 秒闭环** | | | **~4 小时** |
| **30 镜头正片** | | | **~15 小时** |

### 6.2 Ken Burns 方案（24GB 即可，推荐）

| 阶段 | 数量 | 单件耗时 | 合计 |
|---|---|---|---|
| 关键帧 | 8 张 | 4 min | 32 min |
| Ken Burns 动效 | 8 个 | 3s | 24s |
| 配音 | 8 条 | 30s | 4 min |
| 合成 | 1 | 2 min | 2 min |
| **40 秒闭环** | | | **~40 min** |
| **30 镜头正片** | | | **~2.5 小时** |

## 7. 已知问题与回退

| 问题 | 原因 | 解法 |
|---|---|---|
| `Float8_e4m3fn not supported` | MPS 无 fp8 | 换 GGUF 或 fp16 权重 |
| 生成极慢 / CPU 占用高 | 部分算子 MPS 未实现 | `PYTORCH_ENABLE_MPS_FALLBACK=1` 已设；接受速度 |
| OOM / 系统卡死 | 24GB 不够 | 降分辨率到 512×896；关浏览器等内存大户 |
| `mat size error` (umt5 bf16) | 某些 bf16 TE 与 Wan2.2 不兼容 | 用 fp16 版本或 GGUF Q8 |
| ComfyUI 启动报 `comfy_aimdo` | 依赖未装全 | `pip install comfy_aimdo comfy-kitchen alembic` |
| GGUF CLIP 节点不存在 | ComfyUI-GGUF 版本旧 | 更新到最新；或用 bf16 TE + lowvram |

## 8. 不要做的事

- 不要用任何 `fp8` / `fp8_scaled` / `fp8_e4m3fn` 文件
- 不要尝试 14B 视频模型（内存不够 + 速度不可接受）
- 不要尝试 S2V 口型（MPS 不支持）
- 不要用 `--use-pytorch-cross-attention`（MPS 上可能崩溃）
- 不要在 24GB 机器上同时加载图像 TE + UNet（必须顺序）
- 不要期望实时视频生成（MPS 比 CUDA 慢 5-10 倍）

## 9. 参考来源

- [Qwen-Image on Mac: ~13s/step on M4 Pro](https://modelfit.io/blog/qwen-image-2-1-mac-apple-silicon/)
- [FP8 fails on Metal, GGUF Wan2.2 82min/clip (M1 Max 64GB)](https://lilting.ch/en/articles/ltx2-wan22-mac-local-video-gen)
- [Wan2.2 Apple Silicon issue #9255](https://github.com/Comfy-Org/ComfyUI/issues/9255)
- [ComfyUI + MLX on Apple Silicon](https://medium.com/@michael.hannecke/comfyui-mlx-on-apple-silicon-the-clean-native-setup-9dfaa9932a4d)
- [qwen-image-macos native app](https://github.com/zsxkib/qwen-image-macos)
- [Wan2.2 fp16 TE download](https://huggingface.co/Comfy-Org/Wan-2.2/blob/main/split_files/text_encoders/umt5_xxl_fp16.safetensors)
- [ComfyUI Wan2.2 official tutorial](https://docs.comfy.org/tutorials/video/wan/wan2_2)