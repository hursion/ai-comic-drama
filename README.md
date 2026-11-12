# AI 漫剧 · 端到端工作流

面向 **魔搭 ModelScope Notebook / AMD Instinct MI300X（192GB 显存，100 小时额度）** 的 AI 漫剧一次性验证方案。
同时提供 **NVIDIA 24GB 显存 / 32GB 内存 / CUDA 13.x** 的降级兼容配置（GGUF Q4 量化 + Wan2.2 5B），详见 [docs/COMPAT_CUDA24G.md](docs/COMPAT_CUDA24G.md)。

目标不是“堆一个好看的 Demo”，而是在有限额度内**按 Gate 逐段验证、随时可回退、最终产出一集可发布的竖屏漫剧**。

## 先看这三条

1. **先花 8–14 小时打通最小闭环**（1 个角色、8 个镜头、约 40 秒），再决定是否投入 14B 成片。不要第一天就批量跑 30 个镜头。
2. **100GB 持久盘是比 192GB 显存更硬的约束**。主模型统一用 fp8/分时下载；5B 与 14B 不要同时常驻。
3. **ROCm 上唯一需要现场验证的三件事**：fp8 权重能否正常出图、CosyVoice2 依赖能否装通、Wan2.2-S2V 对中文口型是否可用。其余链路风险较低。

## 推荐主链路

```text
剧本/分镜 (Qwen3-4B / 8B)
  -> 角色三视图 + 场景主视觉 (Qwen-Image 20B T2I，一次性)
  -> 每个镜头 1–2 张关键帧 (Qwen-Image-Edit-2511, 2 张参考图)
  -> 图生视频 (Wan2.2 TI2V-5B 草稿 / I2V-A14B + LightX2V 4-step 成片)
  -> 配音 (CosyVoice2-0.5B, zero-shot 音色克隆)
  -> 口型 (Wan2.2-S2V，可选；失败则用口型开合/侧脸规避)
  -> 插帧/超分 (RIFE / Real-ESRGAN)
  -> 字幕、BGM、混音、合成 (ffmpeg)
```

## 目录

| 路径 | 作用 |
|---|---|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 主手册：环境、模型、搭建命令、流水线、Gate、100h 排期、ROCm 踩坑 |
| [docs/COMPAT_CUDA24G.md](docs/COMPAT_CUDA24G.md) | CUDA 24GB / 32GB 内存兼容手册：GGUF 量化、5B 视频、部署命令 |
| [docs/MODELS.md](docs/MODELS.md) | 模型矩阵：体积、许可、ROCm 风险、回退 |
| [docs/CHECKLIST.md](docs/CHECKLIST.md) | 开工/收工检查表与 Gate 验收记录模板 |
| [docs/JUPYTER.md](docs/JUPYTER.md) | JupyterLab、端口映射、会话保活、断点续跑 |
| [notebooks/00_setup.ipynb](notebooks/00_setup.ipynb) | 环境预检 → 安装 → 下载 → 启动 ComfyUI |
| [notebooks/01_episode.ipynb](notebooks/01_episode.ipynb) | 分镜 → 关键帧 → 视频 → 配音 → 合成 |
| [scripts/](scripts/) | 预检、安装、模型下载、ComfyUI 启动、保活 |
| [pipeline/](pipeline/) | 分镜生成、ComfyUI API 批处理、TTS、ffmpeg 合成 |
| [workflows/](workflows/) | 工作流契约、节点命名规范、`extra_model_paths.yaml` |
| [episodes/ep01/episode.example.json](episodes/ep01/episode.example.json) | 8 镜头示例数据契约 |
| [assets/README.md](assets/README.md) | 角色三视图、场景、参考音色的命名与要求 |

## 快速开始（实例内）

```bash
# 1. 把本目录放到 /mnt/workspace/ai-comic-drama（持久盘）
cd /mnt/workspace/ai-comic-drama

# 2. 预检：GPU / ROCm / 磁盘 / 网络
bash scripts/00_preflight.sh

# 3. 安装 ComfyUI + 最小自定义节点（不覆盖系统 torch）
bash scripts/01_bootstrap.sh

# 4. 激活项目环境（后续所有 python 命令都在 venv 里跑）
source venv/bin/activate

# 5. 下载“最小闭环”模型（约 92GB，含一次性 T2I）
python scripts/02_download_models.py --groups image,image-t2i,video-5b,tts,llm-small,asr

# 6. 启动 ComfyUI
bash scripts/03_start_comfyui.sh
curl -s http://127.0.0.1:8188/system_stats | python -m json.tool

# 7. 可选：CosyVoice2 独立环境（不做配音可跳过）
bash scripts/05_install_cosyvoice.sh

# 8. 在 Jupyter 中打开 notebooks/00_setup.ipynb 继续
```

> 角色三视图/场景确认后，可以删掉 20.4GB 的 T2I 主模型（`qwen_image_fp8_e4m3fn.safetensors`），
> 后面关键帧都用 2511 编辑。14B 正片模型等 8 镜头闭环通过后再下。

### CUDA 24GB 实例（8 核 / 32GB 内存）

同一套代码，换用 `cuda24g` profile 即可在 24GB 显存上跑降级链路：

```bash
source configs/cuda24g.env          # 设置 GGUF / lowvram / 分辨率等环境变量
bash scripts/00_preflight.sh
bash scripts/01_bootstrap.sh        # 自动装 cu130 torch + ComfyUI-GGUF
source venv/bin/activate
python scripts/02_download_models.py --profile cuda24g   # ~64GB，GGUF Q4 + 5B
bash scripts/03_start_comfyui.sh    # 自动检测显存 → --lowvram
```

完整差异、模型预算、工作流导出要点见 [docs/COMPAT_CUDA24G.md](docs/COMPAT_CUDA24G.md)。

## 时间预算（结论）

| 方案 | 内容 | 实际 GPU 时长 | 留给你试错的余量 |
|---|---|---|---|
| A. 最小闭环 | 1 角色 / 8 镜头 / 40 秒 | 10–14h | ~86h |
| B. 完整一集 | 30–36 镜头 / 3–4 分钟 / 2 轮迭代 | 30–45h | ~55h |
| C. 冲质量 | 14B 20 步 + 局部重跑 + 多版本 | 55–75h | ~25h |

**建议执行 A→B**：先用 A 打通并记录实测耗时，再用 B 产出正片，剩余额度留给重跑和风格调整。若 A 已消耗超过 20h，直接降级为 5B 全集，不要硬上 14B。

## 重要提醒

- 本仓库的代码与命令是**为 MI300X/ROCm 设计的执行脚手架**，但当前编写环境没有 AMD GPU，**未在真实实例上端到端跑过**。每个阶段都设置了 Gate 和回退方案，请按 Gate 推进，不要跳过 smoke test。
- 魔搭免费实例通常 **1 小时无操作自动关机**；关闭浏览器是否立即释放实例在不同实例类型上行为不一致，**离开前务必手动点“关闭实例”**。
- Qwen-Image-2.1 使用 **Qwen Research License（非商用）**；本方案主链路使用 Apache-2.0 的 Qwen-Image 20B T2I + Qwen-Image-Edit-2511。发布前仍需自行确认输出内容与平台合规要求。
