# 模型矩阵、体积、许可与 ROCm 风险

> 体积以 2026-11 的公开仓库为准，下载前用脚本校验实际大小。**100GB 是持久盘硬上限**，下文的 Set A/B 二选一。

## 1. 主链路模型

| 角色 | 模型 / 文件 | 关键文件与体积 | 许可 | ROCm 风险 | 回退 |
|---|---|---|---|---|---|
| 剧本 / 分镜 LLM | Qwen3-8B / 4B | 8B bf16 ≈ 16GB；4B ≈ 8GB | Apache-2.0 | 低（transformers 即可） | 4B；或云端 API |
| 图像生成（一次性） | Qwen-Image 20B T2I | `qwen_image_fp8_e4m3fn.safetensors` 20.4GB（复用 2511 的文本编码器与 VAE） | Apache-2.0 | 中（fp8 权重） | 2511 空白图编辑兜底；或外部生成三视图 |
| 角色 / 关键帧 / 编辑 | Qwen-Image-Edit-2511 | `qwen_image_edit_2511_fp8mixed.safetensors` 20.5GB；`qwen_2.5_vl_7b_fp8_scaled.safetensors` 9.38GB；`qwen_image_vae.safetensors` 0.25GB；合计 ≈ 30.1GB | Apache-2.0 | 中（fp8 权重） | bf16 主模型 40.8GB；或 SDXL + IP-Adapter |
| 图像加速 LoRA | Qwen-Image-Edit-2511 Lightning 4-step | ≈ 1–2GB | 见模型页（通常 Apache-2.0） | 低 | 40 步 / CFG 4 |
| 视频草稿 | Wan2.2 TI2V-5B | `wan2.2_ti2v_5B_fp16.safetensors` 10.0GB；`umt5_xxl_fp16.safetensors` 11.37GB；`wan2.2_vae.safetensors` 1.41GB；合计 ≈ 22.8GB | Apache-2.0 | 中低 | 直接上 14B；或 Wan2.1 1.3B |
| 视频成片 | Wan2.2 I2V-A14B | high/low 各 `wan2.2_i2v_*_14B_fp8_scaled.safetensors` 14.29GB ×2；`umt5_xxl_fp8_e4m3fn_scaled.safetensors` 6.74GB；`wan_2.1_vae.safetensors` 0.25GB；合计 ≈ 37.9GB | Apache-2.0 | 中（fp8） | bf16 high/low 28.58GB ×2；或 5B |
| 视频加速 LoRA | Wan2.2 LightX2V 4-step | high/low 各 1.23GB；合计 2.46GB | Apache-2.0（见仓库） | 低 | 20 步 / CFG 3.5 |
| 音频驱动 / 口型 | Wan2.2-S2V-14B | `wan2.2_s2v_14B_fp8_scaled.safetensors` 16.4GB；`wav2vec2_large_english_fp16.safetensors` 0.63GB；`umt5_xxl_fp8_e4m3fn_scaled` 6.74GB；`wan_2.1_vae` 0.25GB；合计 ≈ 24GB | Apache-2.0 | 中 | 不做口型；MuseTalk；口型开合 |
| 配音 TTS | CosyVoice2-0.5B | 全量 ≈ 2GB | Apache-2.0 | 中（`pynini` / onnx 依赖） | CPU 推理；Edge-TTS（在线） |
| ASR / 字幕对齐 | SenseVoiceSmall（FunASR） | ≈ 1GB | 见模型页 | 低 | Whisper large-v3；直接用 TTS 时长 |
| 插帧 | RIFE（ComfyUI-Frame-Interpolation） | 模型 ≈ 数十 MB | MIT | 低 | ffmpeg `minterpolate` |
| 超分 | Real-ESRGAN anime 6B | ≈ 17MB | BSD-3-Clause | 低 | ffmpeg `lanczos` |

## 2. 仅限研发 / 不建议商用

| 模型 | 说明 |
|---|---|
| Qwen-Image-2.1 | 7B 视觉生成，2K、RGBA、最多 16 张参考图，统一生成+编辑；**Qwen Research License，非商用**；ComfyUI 文件以 int8 `convrot` 为主，ROCm 自定义量化核风险偏高。适合做效果对比，不建议作为商用主模型。 |
| Qwen-Image-2.1 Turbo / 社区 int8 | 速度更快，但许可与数值稳定性都要逐项确认。 |

## 3. 存储组合（关键）

### Set A：最小闭环（推荐先下）

```text
Qwen-Image-Edit-2511 fp8 三件套   30.1GB
Qwen-Image 20B T2I fp8            20.4GB   # 一次性，资产确认后可删
Wan2.2 TI2V-5B 三件套             22.8GB
CosyVoice2                         2.0GB
Qwen3-4B                           8.0GB
SenseVoiceSmall                    1.0GB
代码 / 缓存 / 输出                 8.0GB
----------------------------------------
合计                             ≈ 92.3GB  ⚠️ 可放，但必须删 T2I 后再堆输出
```

资产确认后删除 `qwen_image_fp8_e4m3fn.safetensors`（20.4GB），占用回到约 72GB；
之后如需 Qwen3-8B 写分镜，再 +8GB，仍在 80GB 左右。

> 不想下 T2I 时，可先试“空白图 + 2511 编辑”生成三视图。这是待验证兜底，
> 崩脸/出空白就回到 20B T2I。

### Set B：正片质量（14B）

```text
Qwen-Image-Edit-2511 fp8 三件套   30.1GB
Wan2.2 I2V-A14B fp8 + LoRA        40.4GB
CosyVoice2                         2.0GB
Qwen3-4B（或临时用 8B）            8.0GB
代码 / 缓存 / 输出                10.0GB
----------------------------------------
合计                             ≈ 90.5GB  ✅ 勉强可放；缓存要控
```

### 不要同时常驻

```text
Set A（含 T2I）+ 14B 视频 ≈ 132GB  ❌ 超 100GB
```

处理办法（按优先级）：

1. **5B 放临时盘**：`/root/models` 或 `/tmp/models`（实例重启会丢，但一次会话够用）。持久盘只放 14B、图像模型和输出。
2. **分时下载**：先用 5B 做草稿，确认镜头后再删 5B、下 14B；用 `scripts/02_download_models.py --groups video-14b`。
3. **LLM/ASR 放临时盘**：故事板生成是一次性任务，Qwen3-8B 用完即可删。
4. **缓存重定向**：`MODELSCOPE_CACHE`、`HF_HOME`、`MIOPEN_USER_DB_PATH`、`TRITON_CACHE_DIR` 全部指向同一个持久目录，避免重复下载/重复调优。

## 4. 显存预算（192GB，实际不是瓶颈）

| 同时加载 | 估算 |
|---|---|
| Qwen-Image-Edit-2511 fp8 + 文本编码器 + VAE | 35–45GB |
| Wan2.2 I2V-A14B high+low fp8 + 文本编码器 + VAE | 45–55GB |
| Wan2.2-S2V-14B fp8 + wav2vec2 + 文本编码器 + VAE | 30–40GB |
| Qwen3-8B bf16 | 16–20GB |

ComfyUI 默认会缓存模型；在 192GB 上可以 `--highvram`，但跨阶段（图像 → 视频 → S2V）切换时仍建议重启 ComfyUI 或让 smart memory 自行卸载，避免“显存没满但碎片化”导致的偶发 OOM。

## 5. 许可与商用

| 资产 | 许可结论 |
|---|---|
| Qwen-Image-Edit-2511 | Apache-2.0，可商用（仍需遵守内容合规） |
| Wan2.2 / Wan2.2-S2V | Apache-2.0 |
| Qwen3 | Apache-2.0 |
| CosyVoice2 | Apache-2.0 |
| Qwen-Image-2.1 | Qwen Research License，**非商用** |
| 字体 / BGM / 音效 | 必须单独确认商用授权；不要随手用影视原声 |

> 发布前建议：保留模型版本、Seed、Prompt、工作流 JSON、生成日期，给平台做 AI 生成内容标识；人物肖像、配音音色需有授权。
