# CUDA 24GB 兼容配置（8 核 / 32GB RAM / 24GB VRAM / Ubuntu 22.04 / CUDA 13.x）

> 结论：**可以跑 AI 漫剧闭环，但不能照搬 MI300X 192GB 的主链路。**
> 必须改成：GGUF Q4 图像模型 + Wan2.2 5B 低分辨率视频 + 顺序加载 + 不做 14B / S2V。

本 profile 已集成到仓库：

- 配置：[configs/cuda24g.env](../configs/cuda24g.env)
- 下载：`python scripts/02_download_models.py --profile cuda24g`
- 启动：`scripts/03_start_comfyui.sh` 会自动检测 24GB 显存并切到 `--lowvram`
- 工作流差异：[workflows/README.md §4.6](../workflows/README.md)

---

## 1. 适用性判断

| 原 MI300X 方案 | 24GB / 32GB 是否适用 | 替代 |
|---|---|---|
| Qwen-Image 20B T2I fp8（20.4GB） | 不建议直接跑 | Qwen-Image GGUF Q4_K_M（12.84GB） |
| Qwen-Image-Edit-2511 fp8（20.5GB） | 不建议直接跑 | 2511 GGUF Q4_K_M（12.8GB） |
| Qwen2.5-VL-7B fp8 文本编码器（9.38GB） | 可用，顺序加载 | 保留原生 fp8，不用 GGUF CLIP |
| Wan2.2 TI2V-5B fp16（10GB） | 可用，降分辨率/帧数 | 480×832 / 81f / 16fps 起步 |
| UMT5-XXL fp16（11.37GB） | 太重 | 换 fp8 scaled（6.74GB） |
| Wan2.2 I2V-A14B fp8 high+low | 不建议 | 留在 5B |
| Wan2.2-S2V-14B | 不建议 | 不做口型，用侧脸/背影/切镜 |
| Qwen3-8B bf16（16GB） | 可跑但挤占显存 | Qwen3-4B（8GB），或云端 API |
| CosyVoice2-0.5B | 可用 | GPU 或 CPU；独立 venv |
| RIFE / Real-ESRGAN | 可用 | 低分辨率插帧/超分，最后做 |

**硬约束不是 CUDA 版本，而是 24GB 显存 + 32GB 内存。**
图像阶段权重约 22.4GB（GGUF 12.8 + TE 9.38 + VAE 0.25），视频阶段约 18.2GB；
两者不能同时常驻，ComfyUI 必须顺序卸载。32GB RAM 刚好够，不要同时开 LLM / TTS / 浏览器大量标签。

---

## 2. CUDA 13 / PyTorch

- `nvcc` 不是必须的；PyTorch wheel 自带 CUDA runtime。
- CUDA 13.0 驱动可以跑 cu130 wheel；如果 cu130 安装或运行异常，退回 cu128。
- 当前稳定线提供 cu130 wheel（例如 torch 2.14.1 / torchvision 0.29.1）。以官网实际版本为准。

手动安装（bootstrap 也会在 `INSTALL_TORCH=1` 时自动做）：

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

# 如果 cu130 不稳定：
# python -m pip install torch torchvision torchaudio \
#   --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
PY
```

通过标准：`cuda True`，设备名正确，显存约 24GB。

---

## 3. 模型与磁盘预算

`--profile cuda24g` 下载：

| 文件 | 大小 | 用途 |
|---|---:|---|
| `qwen-image-edit-2511-Q4_K_M.gguf` | 12.8GB | 关键帧编辑 |
| `qwen-image-Q4_K_M.gguf` | 12.84GB | 一次性三视图/场景 |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | 9.38GB | 图像文本编码器 |
| `qwen_image_vae.safetensors` | 0.25GB | 图像 VAE |
| `wan2.2_ti2v_5B_fp16.safetensors` | 10.0GB | 视频草稿/正片 |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.74GB | 视频文本编码器 |
| `wan2.2_vae.safetensors` | 1.41GB | 5B VAE |
| CosyVoice2-0.5B | 2.0GB | 配音 |
| Qwen3-4B | 8.0GB | 分镜 |
| SenseVoiceSmall | 1.0GB | 可选 ASR |
| **模型合计** | **≈64.4GB** | |
| 代码 / 缓存 / 输出 | ≈8GB | |
| **总占用** | **≈72GB** | 建议磁盘 ≥75GB |

如果磁盘更紧，用 `--profile cuda24g-min`：不下 T2I GGUF，省 12.84GB；
三视图可先试 2511 空白图编辑兜底，或外部生成后导入。

资产确认后也可以删除 `qwen-image-Q4_K_M.gguf`，后面只用 2511 编辑。

---

## 4. 部署命令

```bash
# 本地或云实例均可；路径自定
export WORK=$HOME/ai-comic-drama
cd "$WORK"

# 加载 24GB 兼容 profile
source configs/cuda24g.env

# G0：GPU / CUDA / 磁盘 / 字体
bash scripts/00_preflight.sh

# G1：venv + torch（如缺失）+ ComfyUI + ComfyUI-GGUF
bash scripts/01_bootstrap.sh
source venv/bin/activate

# G2：下载 cuda24g 模型集
python scripts/02_download_models.py --profile cuda24g
python scripts/02_download_models.py --profile cuda24g --verify

# 启动 ComfyUI：自动 --lowvram --reserve-vram 1.5 --cache-none
bash scripts/03_start_comfyui.sh
curl -s http://127.0.0.1:8188/system_stats | python -m json.tool

# 可选：CosyVoice2
bash scripts/05_install_cosyvoice.sh
```

> `configs/cuda24g.env` 默认 `WORK=$HOME/ai-comic-drama`。
> 如果在魔搭 CUDA 实例上，也可以 `export WORK=/mnt/workspace/ai-comic-drama` 后再 source。

---

## 5. 工作流导出差异

仍然按 [workflows/README.md](../workflows/README.md) 的标题契约改名：
`POSITIVE / NEGATIVE / REF1 / REF2 / START_IMAGE / SAVE`。

但 24GB profile 要注意：

1. 安装 `ComfyUI-GGUF`（bootstrap 已按 `INSTALL_GGUF=1` 处理）。
2. GGUF 模型放 `models/comfy/unet/`。
3. 把模板里的 diffusion loader 换成 **`Unet Loader (GGUF)`**。
4. 文本编码器继续用原生 fp8 `qwen_2.5_vl_7b_fp8_scaled.safetensors`，VAE 用 `qwen_image_vae.safetensors`。
5. **第一次不要挂 Lightning 4-step LoRA**；GGUF + LoRA 要单独验证。
6. VAE 解码尽量用 `VAEDecodeTiled`，降低 1080×1920 解码峰值。
7. 视频工作流用 5B fp16 + fp8 UMT5；分辨率/帧数先在 UI 里固定。

导出文件名可以不变，例如 GGUF 编辑工作流仍保存为 `qwen_edit_2511_edit.api.json`。

---

## 6. 推荐生成参数

| 阶段 | 分辨率 | 步数 / CFG | 帧数 / fps | 备注 |
|---|---|---|---|---|
| 三视图 / 场景 | 720×1280 或 1024×1536 | 20 / 4.0 | — | GGUF Q4，先不挂 LoRA |
| 关键帧草稿 | 720×1280 | 20 / 4.0 | — | 2 参考图 |
| 关键帧正片 | 1080×1920 | 28–40 / 4.0 | — | Tiled VAE |
| 5B 视频草稿 | 480×832 | 20 / 5.0 | 81f / 16fps | 约 5 秒 |
| 5B 视频提质 | 720×1280 | 20 / 5.0 | 81f / 16fps | 可能慢/OOM，先测 1 镜 |
| 插帧 | 480p → 720p | — | 16 → 30fps | RIFE 或 ffmpeg |
| 超分 | 720p → 1080p | — | — | Real-ESRGAN anime，最后做 |

`run_batch` 会读取 profile 环境变量：

```bash
python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage sheets
python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage images --limit 2
python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage videos --limit 2
```

如果没 source profile，也可以显式传：

```bash
python -m pipeline.run_batch ... --stage images --steps 20 --cfg 4.0 --width 720 --height 1280
```

---

## 7. 显存 / 内存管理规则

1. **顺序跑阶段**：图像 → 视频 → TTS → 合成。不要同时开两个 ComfyUI 批处理。
2. 图像批处理结束后，如果视频 OOM，重启 ComfyUI：
   ```bash
   kill $(cat logs/comfyui.pid) || true
   bash scripts/03_start_comfyui.sh
   ```
3. Qwen3-4B 分镜、CosyVoice 配音尽量在 ComfyUI 空闲时跑，或直接用 CPU。
4. 保留 `--reserve-vram 1.5`，给桌面/驱动/碎片留余量。
5. `--cache-none` 会降低节点输出缓存，减少 32GB RAM 压力；同批重复镜头会稍慢。
6. 视频 OOM 时按顺序降：
   - 121f → 81f → 49f；
   - 720×1280 → 480×832；
   - 20 步 → 12 步；
   - 关闭其他占显存进程。

---

## 8. Gate 与时间预期（24GB 版）

| Gate | 目标 | 通过标准 | 回退 |
|---|---|---|---|
| G0 | CUDA / 磁盘 / 字体 | torch cuda True，24GB，磁盘 ≥75GB | cu128 / 清盘 |
| G1 | ComfyUI + GGUF 节点 | `/system_stats` 正常，无 CUDA error | 重装节点 / 降 ComfyUI 版本 |
| G2 | 模型完整性 | `--verify` 通过 | 单文件重下 |
| G3 | GGUF 图像 smoke | 1 张三视图 + 1 张关键帧，无崩脸 | Q2_K / 降分辨率 / 外部三视图 |
| G4 | 5B 视频 smoke | 480×832 / 81f 可出片，无全屏噪点 | 49f / 降步数 / 检查 VAE |
| G6 | 8 镜头闭环 | 40 秒成片，角色不跳脸 | 减镜头 / 静帧+运镜 |
| G10 | 配音 | 无爆音，音色统一 | SFT 内置音色 / CPU / 字幕版 |
| G11 | 成片 | 720p 或超分 1080p，音画同步 | 局部重跑 |

时间预期比 MI300X 慢很多：

- GGUF Q4 图像：约 1–3 分钟/张（分辨率/步数影响大）；
- 5B 480p 81f：约 5–15 分钟/段；
- 8 镜头闭环：约 2–5 小时；
- 30 镜头正片：约 20–40 小时，含重跑。

**不要第一天全量。** 先 G3/G4 各 1 个，再 G6 8 镜头。

---

## 9. 明确不建议做的事

- 不要下载 `video-14b`：high+low fp8 权重 + 文本编码器超过 35GB，24GB/32GB 非常勉强。
- 不要下载 `s2v`：14B S2V + wav2vec2 + 文本编码器，口型收益不确定，显存/内存风险高。
- 不要直接跑 `image-bf16`：40.8GB 权重，32GB RAM 都放不下。
- 不要在 GGUF 上第一次就挂 Lightning LoRA：先验证基础出图。
- 不要把 5B 文本编码器用 fp16（11.37GB）：24GB 下优先 fp8（6.74GB）。

---

## 10. 已验证 / 未验证

仓库内已做：

- `--profile cuda24g` 清单、大小、去重；
- shell / Python 语法；
- ComfyUI 启动脚本按显存自动选择 lowvram；
- GGUF 目录 `unet` 与 `extra_model_paths.yaml`；
- notebook 按 `MODEL_PROFILE` 切换图像步数默认值。

**未在真实 24GB CUDA 卡上端到端跑过。**
GGUF Q4 画质、ComfyUI-GGUF 与当前 ComfyUI 版本兼容性、CUDA 13 wheel 稳定性、5B fp8 文本编码器效果，都需要现场 G3/G4 验证。

---

## 11. 参考

- PyTorch CUDA 13.0 stable wheels: <https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325>
- PyTorch previous versions: <https://pytorch.org/get-started/previous-versions/>
- ComfyUI-GGUF: <https://github.com/city96/ComfyUI-GGUF>
- unsloth Qwen-Image-Edit-2511-GGUF: <https://www.modelscope.cn/models/unsloth/Qwen-Image-Edit-2511-GGUF>
- unsloth Qwen-Image-GGUF: <https://huggingface.co/unsloth/Qwen-Image-GGUF>
- ComfyUI Qwen-Image-Edit-2511: <https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511>
- ComfyUI Wan2.2: <https://docs.comfy.org/tutorials/video/wan/wan2_2>
- Unsloth Qwen-Image local tutorial: <https://unsloth.ai/docs/models/tutorials/qwen-image-2512>