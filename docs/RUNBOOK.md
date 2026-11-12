# 魔搭 MI300X · AI 漫剧端到端搭建与一次性验证手册

> 适用环境：魔搭 ModelScope Notebook，AMD Instinct MI300X（192GB HBM3），23 vCPU / 200GB RAM，Ubuntu 22.04，ROCm 7.2 + PyTorch 2.10，100GB 持久化 `/mnt/workspace`，100 小时 GPU 额度。
>
> 本手册的目标：**在 100 小时内，按 Gate 一次性验证并跑通“剧本 → 分镜 → 角色/关键帧 → 图生视频 → 配音/口型 → 合成”的完整链路**，并产出 1 集 3–4 分钟竖屏漫剧。

---

## 0. 结论先行

### 0.1 推荐路线

| 阶段 | 主选 | 备选 | 为什么 |
|---|---|---|---|
| 分镜 | Qwen3-8B（本地） | 云端 LLM / Qwen3-4B | 结构化 JSON 输出，方便批处理 |
| 角色/关键帧 | Qwen-Image-Edit-2511（2 张参考图） | Qwen-Image-2.1（仅研发）/ SDXL+IP-Adapter | 2511 人物一致性强、Apache-2.0、ComfyUI 原生 |
| 视频草稿 | Wan2.2 TI2V-5B，720p / 121 帧 / 24fps | Wan2.1 1.3B | 快，先验证镜头语言 |
| 视频成片 | Wan2.2 I2V-A14B + LightX2V 4-step，720p / 81 帧 / 16fps | 20 步 CFG 3.5 | 4-step 约 4–5× 加速 |
| 配音 | CosyVoice2-0.5B zero-shot | Edge-TTS（在线）/ CPU | 中文自然、可克隆音色 |
| 口型 | Wan2.2-S2V-14B | 不做口型 / MuseTalk | 官方 ComfyUI 工作流，但中文动漫脸需实测 |
| 后期 | RIFE → Real-ESRGAN → ffmpeg | 直接 ffmpeg 超分 | 30/60fps、1080×1920、字幕/混音 |

### 0.2 三条铁律

1. **先跑最小闭环，再批量。** 8 个镜头打通之前，不要下载 14B、不要生成 30 个镜头。
2. **存储优先于显存。** 100GB 持久盘放不下“5B + 14B + 两套图像模型”。用 fp8、分时下载、临时盘。
3. **每个阶段都有 Gate 和时间盒。** 超过时间盒仍不通过，立即切回退，不要在现场“调参到天亮”。

### 0.3 100 小时怎么花

| 方案 | 内容 | GPU 时长 | 余额 |
|---|---|---|---|
| A 最小闭环 | 1 角色 / 8 镜头 / 40 秒 | 10–14h | ~86h |
| B 完整一集 | 30–36 镜头 / 3–4 分钟 / 2 轮迭代 | 30–45h | ~55h |
| C 冲质量 | 14B 20 步 + 局部重跑 + 多版本 | 55–75h | ~25h |

**推荐：A → B。** 先用 A 把“能出片”变成确定性，再用 B 做正片；C 只在 B 的画面仍不达标时启用。

---

## 1. 环境事实与开工前检查

### 1.1 实例规格（现场必须复核）

| 项目 | 预期 | 复核命令 |
|---|---|---|
| GPU | AMD Instinct MI300X，192GB HBM3，gfx942 | `rocminfo \| grep -E "Marketing Name\|gfx"` |
| CPU / RAM | 23 vCPU / 200GB | `nproc`、`free -h` |
| OS | Ubuntu 22.04 | `lsb_release -a` |
| ROCm | 7.2.x，`/opt/rocm` | `cat /opt/rocm/.info/version` |
| PyTorch | 2.10.x + HIP | `python -c "import torch; print(torch.__version__, torch.version.hip)"` |
| 持久盘 | `/mnt/workspace`，100GB | `df -h /mnt/workspace` |
| 额度 | 100h | Notebook 页面 |
| 空闲关机 | 约 1 小时无操作 | 平台规则 |
| 端口映射 | 支持自定义端口 | Notebook 右上角“端口映射” |
| Docker | 通常无 / 不建议依赖 | `docker info` |

> 不同批次的镜像可能是 ROCm 6.2 或 7.2。**以现场 `torch.version.hip` 为准**；如果镜像自带的是 ROCm 6.x，优先使用镜像自带 torch，不要盲目升级到 7.2。

### 1.2 三个纪律

**纪律一：不覆盖系统 torch。**

ComfyUI 的 `requirements.txt` 里有 `torch`，自定义节点也可能拉 `torch`。一旦 pip 装了 CUDA 版，`torch.cuda.is_available()` 会变成 False，或者直接报 HIP 库缺失。做法：

- venv 用 `--system-site-packages`，让系统 ROCm torch 可见；
- 安装 requirements 时**排除 torch/torchvision/torchaudio 三行**；
- 用 `PIP_CONSTRAINT` 锁死版本；
- 自定义节点依赖用 `--no-deps` 或手动装，不要让 pip 解析整棵依赖树。

**纪律二：所有要保留的东西放 `/mnt/workspace`。**

模型、缓存、代码、输出、日志、工作流 JSON 全部放持久盘。`/root`、`/tmp` 只在本次会话有效，实例重启即丢。临时盘可以用来放 5B/LLM 这类“可重下”的模型。

**纪律三：先 smoke，再批量。**

每个模型先做 1 次最小推理：1 张图、1 个 5 秒片段、1 句台词。记录耗时、显存峰值、输出是否正常。只有 smoke 通过，才允许进入批处理。

### 1.3 预检

```bash
cd /mnt/workspace/ai-comic-drama
bash scripts/00_preflight.sh | tee logs/preflight_$(date +%Y%m%d_%H%M).log
```

期望看到：

```text
torch 2.10.0+... hip 7.2...
cuda_available True
device AMD Instinct MI300X ... vram_GB 191.x
modelscope 200 ...
```

如果 `cuda_available False`：

1. 检查 `/dev/kfd`、`/dev/dri` 权限：`ls -l /dev/kfd /dev/dri`；
2. 检查是否在 venv 里被装了 CUDA torch：`pip show torch`；
3. 重新指向系统 torch：`python -c "import torch; print(torch.__file__)"`，必要时删掉 venv 里的 torch 目录；
4. 不要设置 `HSA_OVERRIDE_GFX_VERSION`（MI300X 不需要，设了反而可能错）。

### 1.4 会话、额度与“1 小时空闲关机”

- 平台通常会在**约 1 小时无操作**后自动关机。长时间下载/渲染一般不会触发；风险最大的是“人去吃饭，页面挂着但没任务”。
- 关闭浏览器是否立即释放实例，不同实例类型行为不一致：有的文档说“关闭浏览器不会停止”，也有的免费实例会在断连后释放。**不要赌**：保持 Jupyter 标签页打开，跑 `scripts/04_keepalive.py`，离开超过 10 分钟就手动点“关闭实例”。
- 每次收工前：`bash scripts/99_stop_check.sh`（或手动）确认 ComfyUI 已停、实例已关、额度已扣减。
- 持久盘里的模型不会因为关机丢失；下次启动直接复用。

---

## 2. 模型与存储

完整矩阵见 [MODELS.md](MODELS.md)。这里只给执行结论。

### 2.1 两条模型组合

**Set A（先下，含一次性 T2I，约 92GB）：**

```text
Qwen-Image-Edit-2511 fp8 三件套   30.1GB
Qwen-Image 20B T2I fp8            20.4GB   # 只用于角色三视图/场景主视觉
Wan2.2 TI2V-5B 三件套             22.8GB
CosyVoice2                         2.0GB
Qwen3-4B                           8.0GB
SenseVoiceSmall                    1.0GB
代码/缓存/输出                     8.0GB
```

生成并确认 `assets/` 后，可以直接删掉 T2I 主模型（`models/comfy/diffusion_models/qwen_image_fp8_e4m3fn.safetensors`）
腾出 20.4GB；后面所有关键帧都用 2511 编辑。若想用 Qwen3-8B 写分镜，再替换 4B（+8GB），
删掉 T2I 后仍然放得下。

> 如果不想下 T2I，也可以先试 `qwen_edit_2511_t2i.api.json` 的“空白图 + 编辑”兜底。
> 这属于 G3 待验证项，效果不稳定；一旦崩脸就回到 20B T2I，不要硬撑。

**Set B（正片，约 90.5GB）：**

```text
Qwen-Image-Edit-2511 fp8 三件套   30.1GB
Wan2.2 I2V-A14B fp8 + 4-step LoRA 40.4GB
CosyVoice2                         2.0GB
Qwen3-4B（或临时 8B）              8.0GB
代码/缓存/输出                    10.0GB
```

> 不要同时保留 5B 和 14B。Set A（含 T2I）+ 14B ≈ 132GB，必爆。把 5B 放 `/root/models`，或先删 5B 再下 14B。

### 2.2 目录约定

```text
/mnt/workspace/ai-comic-drama/
├── ComfyUI/                    # 代码
├── venv/                       # ComfyUI / 主流程环境（system-site-packages）
├── venv-tts/                   # CosyVoice2 独立环境（可选）
├── CosyVoice/                  # CosyVoice2 代码（可选）
├── models/
│   ├── comfy/                  # ComfyUI 通过 extra_model_paths.yaml 读取
│   │   ├── diffusion_models/
│   │   ├── text_encoders/
│   │   ├── vae/
│   │   ├── loras/
│   │   ├── audio_encoders/
│   │   └── upscale_models/
│   ├── cosyvoice2/
│   ├── qwen3-4b/               # 或 qwen3-8b/
│   └── sensevoice/
├── assets/
│   ├── characters/             # 角色三视图、表情、服装
│   ├── scenes/                 # 场景主视觉
│   └── voices/                 # 3–10 秒音色参考
├── episodes/ep01/
│   ├── episode.json
│   ├── keyframes/
│   ├── clips/
│   ├── audio/
│   └── final/
├── workflows/                  # ComfyUI API 工作流
├── logs/
└── cache/                      # miopen / triton / hf / modelscope
```

### 2.3 缓存重定向（必须做）

```bash
export WORK=/mnt/workspace/ai-comic-drama
export MODELSCOPE_CACHE=$WORK/cache/modelscope
export HF_HOME=$WORK/cache/huggingface
export MIOPEN_USER_DB_PATH=$WORK/cache/miopen
export MIOPEN_FIND_MODE=FAST
export TRITON_CACHE_DIR=$WORK/cache/triton
```

- MIOpen 第一次卷积会做算法搜索，缓存后第二次明显变快；不持久化等于每次重启都重新调优。
- Triton 缓存同理，视频模型的自定义 kernel 很多。
- 缓存目录本身可能 5–15GB，要算进 100GB 预算。

---

## 3. 端到端流水线设计

### 3.1 数据契约：`episode.json`

整条链路只认一个文件：`episodes/ep01/episode.json`。剧本、分镜、参考图、镜头参数、输出路径都在里面。

最小结构：

```json
{
  "title": "夜班电梯",
  "style": "国漫赛璐璐，干净线稿，冷色调，电影感构图，9:16 竖屏",
  "aspect_ratio": "9:16",
  "fps": 30,
  "characters": [
    {
      "id": "lin",
      "name": "林晚",
      "appearance": "25 岁女性，黑色齐肩短发，琥珀色眼睛，右眼下有一颗小痣",
      "outfit": "深蓝色保安制服，银色工牌，黑色短靴",
      "sheet": "assets/characters/lin_sheet.png",
      "voice_ref": "assets/voices/lin.wav"
    }
  ],
  "locations": [
    {
      "id": "elevator",
      "name": "老旧电梯",
      "master": "assets/scenes/elevator_master.png",
      "prompt": "1990 年代老式电梯内部，金属墙面，暖黄顶灯，楼层指示灯，轻微恐怖氛围"
    }
  ],
  "shots": [
    {
      "id": "s01",
      "duration": 5,
      "location": "elevator",
      "characters": ["lin"],
      "shot_size": "medium",
      "camera": "slow push-in",
      "action": "林晚走进电梯，按下 13 楼，门缓缓合上",
      "dialogue": [
        {"character": "lin", "text": "又是最后一班。", "emotion": "疲惫"}
      ],
      "image_prompt": "林晚站在老旧电梯里，右手刚按下楼层按钮，回头看向镜头，冷色顶光，半身构图",
      "video_prompt": "镜头缓慢推近，林晚眨眼，发丝轻微摆动，电梯门在背景缓缓合上，保持人物面部稳定",
      "negative_prompt": "多余手指，肢体畸形，脸部崩坏，文字水印，低清晰度，噪点，闪烁，镜头抖动",
      "refs": ["assets/characters/lin_sheet.png", "assets/scenes/elevator_master.png"],
      "keyframe_start": "keyframes/s01_start.png",
      "clip": "clips/s01.mp4"
    }
  ]
}
```

字段约定：

- `duration`：镜头时长，4–6 秒。漫剧节奏快，超过 8 秒观众会走。
- `shot_size`：`wide` / `medium` / `close-up` / `detail`。
- `camera`：`static` / `slow push-in` / `pull-out` / `pan left` / `handheld` / `orbit`。
- `refs`：Qwen-Image-Edit-2511 的 2 张参考图。第 1 张固定为角色三视图，第 2 张为场景主视觉或上一镜关键帧。
- `image_prompt`：只描述“这一帧”，不要写剧情。
- `video_prompt`：只描述“运动”，不要重复外貌（外貌由参考图锁）。
- `negative_prompt`：全片统一，放在配置里，避免每个镜头漂移。

### 3.2 角色一致性

这是漫剧的生死线。按优先级：

1. **角色三视图（必须）**：正面/侧面/背面 + 3 个表情，白底或浅灰底，统一服装。用 Qwen-Image 20B T2I 生成，人工挑 1 张最稳的作为 `sheet`。
2. **2 参考图编辑（主方案）**：每个镜头把 `sheet` + `场景主视觉` 一起喂给 Qwen-Image-Edit-2511，提示词明确“保持参考图 1 的脸型、发型、瞳色、服装完全一致，只改变表情、姿势和机位”。
3. **固定 Seed + 固定风格前缀**：同一角色同一场景，Seed 不要每次随机；风格前缀写进配置，不要每个镜头改。
4. **上一镜关键帧作为第二参考（进阶）**：连续镜头用上一镜的最后一帧替代场景图，能显著减少跳变；但会放大错误，适合已经稳定的镜头。
5. **LoRA（可选，不建议第一轮做）**：用 20–30 张已确认的角色图训练角色 LoRA。MI300X 有能力，但会吃掉 2–4 小时和额外存储；第一轮用参考图编辑即可，LoRA 留作第二轮提质。

一致性验收：

- 人工看接触表（contact sheet）：脸型、发型、瞳色、服装、配饰五要素是否一致；
- 自动指标（可选）：InsightFace 提取人脸 embedding，计算与 `sheet` 的余弦相似度；低于 0.55 的镜头标记重跑；
- 注意：动漫脸的人脸检测本身不稳定，自动指标只做辅助，最终以人工为准。

### 3.3 分镜与镜头语言

- 3–4 分钟 ≈ 30–36 个镜头 × 5 秒。第一轮先做 24 个镜头，跑通后再补。
- 每个镜头只做**一个动作 + 一个机位**。不要在一个 5 秒镜头里让角色“走进来、坐下、说话、回头”。
- 对话镜头优先 `medium` / `close-up`，给口型留空间；动作镜头用 `wide` / `detail`，口型不明显，可以不做 lip sync。
- 转场：同一场景硬切；跨场景用 0.2–0.3 秒黑场或白场；不要用花哨转场。
- 关键帧：每个镜头 1 张起始帧；需要“从 A 到 B”的镜头再加 1 张结束帧，用 Wan2.2 FLF2V。

### 3.4 关键帧生成（Qwen-Image-Edit-2511）

**模型文件放置：**

```text
# 一次性资产生成（角色三视图/场景）
ComfyUI/models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors

# 关键帧编辑
ComfyUI/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors
ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
ComfyUI/models/vae/qwen_image_vae.safetensors
ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

> 角色三视图和场景主视觉用 **Qwen-Image 20B T2I** 生成一次，之后所有镜头都用 2511 编辑。
> 2511 是编辑模型，官方模板要求输入图；不要把它当纯 T2I 用，除非 G3 已验证空白图兜底可用。

**参数：**

| 模式 | 步数 | CFG | 分辨率 | 用途 |
|---|---|---|---|---|
| 草稿 | 4（Lightning LoRA） | 1.0 | 720×1280 | 快速看构图/一致性 |
| 成片 | 40 | 4.0 | 1440×2560 或 1080×1920 | 最终关键帧 |
| 折中 | 8–12（Lightning） | 1.0–1.5 | 1080×1920 | 第二轮批量 |

> 2511 的 ComfyUI 官方工作流默认用 **2 张参考图**；正好对应“角色三视图 + 场景主视觉”。
> 注意：编辑模型的输出尺寸跟随输入图。工作流里要有一个 `ImageScale` 把参考图统一到
> `1080×1920`（或 720×1280 草稿），否则后期只能加黑边。`--width/--height` 只改 T2I 的 latent。

**提示词模板（图像编辑）：**

```text
保持参考图 1 中角色的脸型、发型、瞳色、右眼下小痣和深蓝色保安制服完全一致；
参考图 2 是场景与光影参考。将角色放入参考图 2 的场景中，生成 9:16 竖屏电影感构图。
镜头：中景，平视，角色位于画面右侧三分之一。
动作/表情：<本镜头动作与表情>。
风格：<全局风格前缀>。
不要改变角色身份，不要添加参考图中不存在的配饰。
```

**负面提示词（全片统一）：**

```text
多余手指，六指，肢体畸形，脸部崩坏，眼睛不对称，双瞳，文字，水印，logo，
低清晰度，模糊，噪点，过曝，塑料皮肤，3D 渲染感，镜头畸变，画面撕裂
```

### 3.5 图生视频（Wan2.2）

**5B 草稿（推荐先跑）：**

- 文件：`wan2.2_ti2v_5B_fp16.safetensors` + `umt5_xxl_fp16.safetensors` + `wan2.2_vae.safetensors`
- 分辨率：1280×704（训练分辨率）或 720×1280 竖屏（需要测试边缘质量）
- 帧数：121 帧 / 24fps ≈ 5 秒；帧数用 `4n+1`
- 步数 / CFG：20 / 5.0（以模板默认为准）
- 用途：镜头语言、构图、运动可行性验证；不追求最终画质

**14B 成片：**

- 文件：`wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` + `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` + `umt5_xxl_fp8_e4m3fn_scaled.safetensors` + `wan_2.1_vae.safetensors`
- 分辨率：720p；帧数 81 帧 / 16fps ≈ 5 秒
- 默认：20 步 / CFG 3.5（高噪 + 低噪两段）
- 加速：LightX2V 4-step LoRA，步数 4 / CFG 1.0；官方实测约 4–5× 加速，代价是动态性略降
- 用途：正片；只对“值得”的镜头开 20 步

**首尾帧（FLF2V）：**

- 用于“同一镜头内从 A 到 B”的过渡，例如角色从门口走到窗边。
- 起始帧和结束帧都用 Qwen-Image-Edit-2511 生成，保持同一角色/场景。
- ComfyUI 官方模板：`Wan 2.2 14B First-Last Frame to Video`。

**视频提示词模板：**

```text
镜头：<static / slow push-in / pull-out / pan left / handheld>。
动作：<一个明确动作，例如“林晚眨眼，右手缓慢按下按钮”>。
环境运动：<发丝轻摆 / 灯光闪烁 / 背景人流虚化 / 电梯门合上>。
保持人物面部、服装、发型与起始帧一致；保持构图稳定；不要出现第二个人。
```

> Wan2.2 对“纯旋转/纯环绕”容易退化成平面旋转；需要环绕时明确写“3D camera orbit”，否则优先用推近/拉远。

### 3.6 配音与口型

**CosyVoice2（主选）：**

- 每个角色准备 3–10 秒干净参考音（无 BGM、无混响），放到 `assets/voices/<id>.wav`。
- 按“一句台词一个 wav”生成，命名 `audio/<shot_id>_<idx>.wav`，方便后期按镜头对齐。
- 情绪通过参考音 + 文本标点控制；CosyVoice2 对感叹号、停顿比较敏感，台词里适当加逗号。
- ROCm 上优先 `load_jit=False, load_trt=False, fp16=False`，稳定优先；失败则切 CPU（0.5B 模型，23 核足够跑几分钟对白）。

**口型（可选）：**

- **Wan2.2-S2V-14B**：一张图 + 一段音频 → 说话视频；官方 ComfyUI 工作流，77 帧 / 16fps 一段，可续接。适合中近景对话。
- **风险**：S2V 的训练数据以真人为主，动漫脸可能出现“五官漂移/口型夸张”。先拿 1 个镜头测试；不行就退回“口型开合 + 侧脸/背影/切镜”规避。
- **MuseTalk / LatentSync**：社区方案，ROCm 未验证；不要把它们放进第一轮关键路径。

**字幕：**

- 最稳的方式：直接用 TTS 的台词和时长生成 ASS，不依赖 ASR。
- 需要自动对齐时用 SenseVoiceSmall / Whisper，但要以人工校对为准。

### 3.7 后期

1. **插帧**：RIFE 2×（16fps → 32fps 或 24fps → 48fps），或 ffmpeg `minterpolate`（慢，质量一般）。
2. **超分**：关键帧用 Qwen-Image 直接出 1080×1920；视频用 Real-ESRGAN anime 2× 后再缩到 1080×1920，或直接 ffmpeg `lanczos`。
3. **统一规格**：1080×1920，30fps，SAR 1:1，H.264，yuv420p。
4. **字幕**：ASS 样式（思源黑体/Noto Sans CJK，字号 54–64，底部安全区 180px，描边 3px）。
5. **混音**：对白 -16 LUFS 左右，BGM -26 ～ -22 LUFS，侧链压缩让 BGM 在对白时自动下沉；最终 `loudnorm=I=-14:TP=-1.5:LRA=11`。
6. **导出**：`-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart`。

### 3.8 QC（每集必做）

| 检查 | 方法 | 不通过怎么办 |
|---|---|---|
| 角色一致性 | 接触表 + 人脸 embedding | 重跑该镜头，换参考图/Seed |
| 画面崩坏 | 逐镜头看首帧/中帧/尾帧 | 降低动作幅度，改近景，重跑 |
| 闪烁/跳变 | 抽帧对比，或看视频 | 换 14B、降 CFG、固定 Seed |
| 口型 | 对白镜头逐句看 | 改侧脸/背影，或不做口型 |
| 字幕 | 人工读一遍 | 修文本，重新生成 ASS |
| 音频 | 看波形/响度 | 重新 TTS 或混音 |
| 时长 | `ffprobe` | 调整镜头时长/删减 |

---

## 4. 搭建步骤（复制即用）

> 以下命令默认在实例内的 `/mnt/workspace/ai-comic-drama` 执行。先把本仓库上传/解压到该目录。

### 4.1 预检

```bash
cd /mnt/workspace/ai-comic-drama
bash scripts/00_preflight.sh | tee logs/preflight.log
```

### 4.2 安装 ComfyUI（ROCm 安全）

```bash
bash scripts/01_bootstrap.sh
```

脚本做四件事：

1. 建目录、写缓存环境变量；
2. 创建 `venv --system-site-packages`，记录系统 torch 版本；
3. clone ComfyUI（默认 `v0.18.2`，失败自动尝试镜像），安装 requirements 时排除 torch 三件套；
4. 安装最小自定义节点：`ComfyUI-Manager`、`VideoHelperSuite`、`KJNodes`、`Frame-Interpolation`。

> 如果 `v0.18.2` 标签不存在，脚本会退回 `master`。记录 `git rev-parse HEAD`，写进实验日志。

手动核对：

```bash
source venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

> 后面所有 `python -m pipeline...` / `python scripts...` 都应在同一个 venv 里执行。
> Jupyter 的 kernel 是系统 Python，`00_setup.ipynb` 会用 `PATH` 把 `!python` 切到 venv；
> 如果你在终端里跑，先 `source venv/bin/activate`。

### 4.3 下载模型

```bash
# Set A：最小闭环（约 92GB，含一次性 T2I）
python scripts/02_download_models.py --groups image,image-t2i,video-5b,tts,llm-small,asr

# 资产确认后，可删 T2I 主模型腾出 20.4GB（可选）
# rm models/comfy/diffusion_models/qwen_image_fp8_e4m3fn.safetensors

# 正片前再下 14B（约 40GB；先确认磁盘，且先删 5B 或放临时盘）
python scripts/02_download_models.py --groups video-14b

# 只下载某一组 / 看清单 / 校验
python scripts/02_download_models.py --list
python scripts/02_download_models.py --groups image --dry-run
python scripts/02_download_models.py --groups image,image-t2i,video-5b,tts,llm-small,asr --verify
```

脚本会：

- 用 ModelScope 单文件下载，避免把整个 repackaged 仓库（数百 GB）拉下来；
- 校验文件大小（±15% 警告）；
- 在 `models/comfy/*` 下建符号链接；
- 已存在的文件跳过。

> 如果某个文件路径变了，脚本会打印实际仓库文件列表和失败项；按提示改 `MANIFEST` 即可，不要直接 `snapshot_download` 整个仓库。

### 4.4 模型路径

ComfyUI 通过 `workflows/extra_model_paths.yaml` 读取 `models/comfy`：

```yaml
comic:
  base_path: /mnt/workspace/ai-comic-drama/models/comfy
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  loras: loras
  upscale_models: upscale_models
```

启动时加 `--extra-model-paths-config`。

### 4.5 启动 ComfyUI

```bash
bash scripts/03_start_comfyui.sh
sleep 5
curl -s http://127.0.0.1:8188/system_stats | python -m json.tool
tail -f logs/comfyui.log
```

脚本默认参数：

```bash
python main.py \
  --listen 0.0.0.0 --port 8188 \
  --use-pytorch-cross-attention \
  --highvram \
  --preview-method none \
  --extra-model-paths-config workflows/extra_model_paths.yaml
```

- `--use-pytorch-cross-attention`：走 PyTorch SDPA / AOTriton，ROCm 上比 xformers 稳。
- `--highvram`：192GB 足够，减少卸载/重载。
- 不要加 `--use-split-cross-attention`、`--disable-smart-memory` 作为默认；出问题再逐个试。
- 不要启用 `torch.compile` 的 `reduce-overhead`（ROCm 上已知会崩）。

### 4.6 Jupyter、端口映射与 API

- **批处理不需要端口映射**：Jupyter 内核直接访问 `http://127.0.0.1:8188`，最稳。
- **要在浏览器里看 ComfyUI UI**：Notebook 页面右上角“端口映射” → 添加 8188 → 用生成的地址访问。不同批次可能是 `https://.../proxy/8188/` 或独立域名；以页面提示为准。
- 端口映射失败时，用 `cpolar`/`ngrok` 之类做临时穿透；仅用于调试，不要把无鉴权的 ComfyUI 暴露在公网。
- Jupyter 操作细节见 [JUPYTER.md](JUPYTER.md)。

### 4.7 导入工作流并统一节点标题

ComfyUI 的 API JSON 不能手写，必须从官方模板导出：

1. 打开 ComfyUI UI → `Workflow` → `Browse Templates`；
2. 图像：`Qwen-Image`（20B T2I，生成三视图/场景）和 `Qwen-Image-Edit-2511`（关键帧编辑）；
3. 视频：选 `Wan 2.2 TI2V 5B`、`Wan 2.2 14B I2V`、`Wan 2.2 14B First-Last Frame`、`Wan 2.2 S2V`；
4. 在画布上把关键节点改名为：

| 标题 | 节点类型 | 被脚本修改的输入 |
|---|---|---|
| `POSITIVE` | `CLIPTextEncode` / `TextEncodeQwenImageEditPlus` | `text` / `prompt` |
| `NEGATIVE` | 同上 | `text` / `prompt` |
| `REF1` | `LoadImage` | `image` |
| `REF2` | `LoadImage` | `image` |
| `START_IMAGE` | `LoadImage` | `image` |
| `AUDIO` | `LoadAudio` | `audio` |
| `SAVE` | `SaveImage` / `VHS_VideoCombine` | `filename_prefix` |

5. `Workflow` → `Export (API)`，保存到 `workflows/`：
   - `qwen_image_t2i.api.json`（角色三视图 + 场景，20B T2I）
   - `qwen_edit_2511_edit.api.json`（关键帧，2 参考图）
   - `qwen_edit_2511_t2i.api.json`（可选兜底：空白图编辑）
   - `wan22_5b_i2v.api.json`
   - `wan22_14b_i2v.api.json`
   - `wan22_14b_flf2v.api.json`
   - `wan22_s2v.api.json`

> 如果节点标题对不上，`run_batch` 会打印工作流中所有标题；按提示改名再导出即可。详见 [workflows/README.md](../workflows/README.md)。

### 4.7.1 分镜（可选）

如果不手写 `episode.json`，可以让本地 Qwen3 生成一版，再人工改：

```bash
# 先备份/复制示例，避免覆盖已改好的分镜
python -m pipeline.storyboard \
  --story episodes/ep01/story.example.md \
  --out episodes/ep01/episode.qwen.json \
  --model models/qwen3-4b \
  --shots 8
```

- 用 4B 时把 `--model` 指到 `models/qwen3-4b`；8B 则 `models/qwen3-8b`。
- 输出只是初稿：镜头时长、参考图、台词都要人工校对，再替换 `episode.json`。
- 解析失败会在同目录留 `.raw.txt`；优先降 `--shots` 或加 `--max-new-tokens`，不要反复抽卡。

### 4.8 批处理

```bash
# 0) 角色三视图 + 场景主视觉（只跑一次）
python -m pipeline.run_batch \
  --episode episodes/ep01/episode.json \
  --stage sheets \
  --workflow workflows/qwen_image_t2i.api.json

# 1) 只看前 2 个镜头，验证 API 链路
python -m pipeline.run_batch \
  --episode episodes/ep01/episode.json \
  --stage images \
  --workflow workflows/qwen_edit_2511_edit.api.json \
  --limit 2 --steps 4 --cfg 1.0

# 草稿视频
python -m pipeline.run_batch \
  --episode episodes/ep01/episode.json \
  --stage videos \
  --workflow workflows/wan22_5b_i2v.api.json \
  --limit 2

# 正片视频（14B + 4-step LoRA 工作流）
python -m pipeline.run_batch \
  --episode episodes/ep01/episode.json \
  --stage videos \
  --workflow workflows/wan22_14b_i2v.api.json
```

### 4.9 配音

CosyVoice2 的依赖（`pynini`、`WeTextProcessing`）容易和 ComfyUI 冲突，所以单独建 `venv-tts`：

```bash
bash scripts/05_install_cosyvoice.sh
source venv-tts/bin/activate
export PYTHONPATH=$PWD/CosyVoice
```

脚本是 best-effort，时间盒 45 分钟。`pynini` 装不上时先试
`conda install -c conda-forge pynini==2.1.5`，仍失败就接受“本轮无配音”，
用字幕 + 环境音先出片（见风险矩阵）。

```bash
# 单句测试
python -m pipeline.tts_cosyvoice \
  --text "又是最后一班。" \
  --prompt-text "这是一段参考音色的示例台词。" \
  --prompt-wav assets/voices/lin.wav \
  --out episodes/ep01/audio/test.wav

# 整集（有参考音时）
python -m pipeline.tts_cosyvoice --episode episodes/ep01/episode.json

# 整集（没有参考音的兜底：用 CosyVoice2 内置音色）
python -m pipeline.tts_cosyvoice --episode episodes/ep01/episode.json --sft-speaker 中文女
```

> `voice_text` 必须是参考音频的**准确转写**，否则 zero-shot 音色会漂。
> 没有参考音时，`--sft-speaker` 只用于跑通链路；正式配音请补 `assets/voices/<id>.wav`。
> 输出按句命名：`episodes/ep01/audio/<shot_id>_<line_index>.wav`，后处理直接按这个顺序对齐。

### 4.10 合成

```bash
# 草稿：720×1280，快速看节奏
python -m pipeline.assemble \
  --episode episodes/ep01/episode.json \
  --draft

# 正片：1080×1920，字幕 + BGM + loudnorm
python -m pipeline.assemble \
  --episode episodes/ep01/episode.json \
  --bgm assets/bgm/theme.mp3
```

输出：

```text
episodes/ep01/final/ep01.mp4
episodes/ep01/final/ep01.srt
episodes/ep01/final/cover.jpg
episodes/ep01/final/manifest.json
```

---

## 5. 100 小时排期（按 Gate 推进）

### 5.1 总表

| # | 阶段 | 时间盒 | 产出 | Gate | 失败回退 |
|---|---|---|---|---|---|
| 0 | 离线准备 | 4–8h | 故事、角色设定、音色参考、风格前缀、脚本 | 物料齐全 | 不占 GPU，可反复 |
| 1 | 环境预检 | 0.5h | `preflight.log` | G0 | 修 torch/权限 |
| 2 | 安装 + 启动 | 1–2h | ComfyUI `/system_stats` | G1 | 换镜像/降版本 |
| 3 | 下载 Set A | 0.5–1.5h | 模型文件 + 大小校验 | G2 | 单文件重下 |
| 4 | 图像 smoke | 1–2h | 1 张 2K 角色图 | G3 | bf16 / 降分辨率 |
| 5 | 视频 smoke | 1–2h | 1 个 5 秒 5B 片段 | G4 | 降分辨率/帧数 |
| 6 | 角色/场景资产 | 2–4h | 三视图 + 3 个场景主视觉 | G5 | 多抽几张 |
| 7 | 8 镜头闭环 | 4–6h | 8 关键帧 + 8 片段 + 配音 + 40 秒成片 | G6 | 回退到 5B/不做口型 |
| 8 | 14B 提质 | 2–3h | 1 个 14B 镜头 + 4-step 对比 | G7 | 留在 5B |
| 9 | 正片关键帧 | 4–8h | 24–36 张 | G8 | 减镜头/降分辨率 |
| 10 | 正片视频 | 6–15h | 24–36 个片段 | G9 | 4-step/5B |
| 11 | 配音/口型 | 2–5h | 全片对白 + 可选 S2V | G10 | 不做口型 |
| 12 | 合成/QC | 2–4h | 3–4 分钟成片 | G11 | 局部重跑 |
| 13 | 缓冲/二轮 | 20–30h | 提质、补镜头、风格统一 | — | — |

### 5.2 五天执行建议

| 天 | 内容 | 目标 |
|---|---|---|
| D0（离线） | 剧本、角色卡、场景卡、音色参考、Prompt 模板 | 到实例就能开工 |
| D1 | 预检 → 安装 → Set A → 图像/视频 smoke | G0–G4 |
| D2 | 角色三视图/场景 → 8 镜头闭环 | G5–G6 |
| D3 | 14B 单镜头对比 → 正片关键帧 | G7–G8 |
| D4 | 正片视频 → 配音/口型 → 合成 | G9–G11 |
| D5 | QC、局部重跑、导出、归档 | 成片 |

> 每天收工前把当日产出、耗时、失败原因写进 `docs/CHECKLIST.md` 的 Run Log。第二天先看日志再开工，避免重复踩坑。

### 5.3 时间盒规则

- 单个 Gate 超过时间盒 **1.5 倍**仍未通过：立即切回退，不要继续调参。
- 单个镜头重跑超过 **3 次**：换构图/换参考图/换模型，不要死磕 Seed。
- 视频批处理时，先跑 3 个镜头测速，再推算全片时间；如果超过当天预算，降分辨率或改 5B。

---

## 6. Gate 验收标准

### G0 环境

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
df -h /mnt/workspace   # 可用 > 95GB
```

通过标准：设备名含 `MI300X`，显存 ≈ 192GB，持久盘可用 > 95GB。

### G1 ComfyUI

```bash
curl -s http://127.0.0.1:8188/system_stats
```

通过标准：返回 JSON，`devices[0].name` 含 `MI300X`；日志无 HIP error。

### G2 模型

```bash
python scripts/02_download_models.py --groups image,image-t2i,video-5b,tts,llm-small,asr --verify
```

通过标准：所有文件大小在预期 ±15%；ComfyUI 能加载模型（不报 missing keys 之外的错）。

### G3 图像 smoke

先分两步，任何一步失败都不要继续：

1. **T2I 一次性**：用 Qwen-Image 20B 生成 1 张角色三视图和 1 张场景主视觉；
2. **2511 编辑**：拿三视图 + 场景生成 1 张 1080×1920 关键帧。

- 记录：耗时、显存峰值、是否崩脸、文字是否乱码；
- 通过标准：单张 < 90 秒（fp8 + 40 步），画面无严重畸形、三视图五要素一致；
- 失败回退：Lightning 4-step 先看构图；bf16 主模型；降分辨率到 720×1280；
  2511 空白图 T2I 跑不通就回 20B T2I，不要硬撑。

### G4 视频 smoke

- 1 个 5 秒 5B 片段，720p；
- 记录：耗时、显存峰值、是否出现“纯彩色噪点”、是否平面旋转；
- 通过标准：单片段 < 8 分钟，画面稳定，无全屏噪点；
- 失败回退：降帧数到 49/81；降分辨率；检查 VAE/文本编码器是否放错目录；检查是否误用 CUDA 版 torch。

> “纯彩色噪点”是 ROCm 上常见的假成功：流程跑完但输出噪声。优先检查模型文件完整性、VAE 是否匹配（5B 用 `wan2.2_vae`，14B 用 `wan_2.1_vae`）、文本编码器是否 fp8 兼容。

### G5 角色/场景资产

- 1 个角色三视图 + 3 个表情；
- 3 个场景主视觉；
- 通过标准：人工确认“可以作为全片参考”；不合格就多抽 4–8 张。

### G6 8 镜头闭环

- 8 个关键帧 + 8 个 5B 片段 + 8 句配音 + 40 秒成片；
- 通过标准：能连续看完，角色不跳脸，字幕/音频对齐；
- 这是**最重要的 Gate**：通过后再投入 14B 和 30 镜头。

### G7 14B 提质

- 同一关键帧，分别用 5B、14B 20 步、14B 4-step 生成；
- 记录三者耗时与画质；
- 通过标准：14B 4-step 在可接受时间内明显优于 5B；否则全片留 5B。

### G8 正片关键帧

- 24–36 张，按镜头命名；
- 通过标准：角色一致性抽查 10 张，至少 8 张合格；否则先修参考图/提示词。

### G9 正片视频

- 24–36 个片段；
- 通过标准：无全屏噪点、无严重闪烁、动作可接受；不合格镜头重跑不超过 3 次。

### G10 配音/口型

- 全片对白 wav；
- 可选：3 个 S2V 对话镜头；
- 通过标准：无爆音、无断句、音色统一；S2V 不崩脸才全片用。

### G11 成片

- 1080×1920 / 30fps / H.264；
- 字幕无错别字、无出框；
- 响度 ≈ -14 LUFS；
- 通过标准：完整看 2 遍，记录需要重跑的镜头清单。

---

## 7. ROCm 踩坑清单

| 现象 | 原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available() == False` | venv 里被装了 CUDA torch | 删 venv 内 torch，重装 requirements 时排除 torch 三件套 |
| `libamdhip64.so` 找不到 | 装错 wheel / 环境变量 | 使用镜像自带 torch；`source /opt/rocm/bin/rocm-env.sh` 视镜像而定 |
| ComfyUI 启动报 xformers 错 | xformers 在 ROCm 上非官方 | 移除 `--enable-xformers`；用 `--use-pytorch-cross-attention` |
| 第一次推理极慢 | MIOpen/Triton 在调优 | 持久化 `MIOPEN_USER_DB_PATH` / `TRITON_CACHE_DIR`；第二次会快 |
| 输出纯彩色噪点 | 模型/VAE/文本编码器不匹配；fp8 支持问题 | 核对文件；5B 用 `wan2.2_vae`，14B 用 `wan_2.1_vae`；切 bf16 |
| fp8 模型报 `_scaled_mm` 错 | ROCm 对 fp8 算子支持不完整 | 换 bf16/fp16 权重；或只对文本编码器用 fp8 |
| int8 `convrot` 报自定义算子错 | 社区量化核非 ROCm 原生 | 不用 Qwen-Image-2.1 int8；换 2511 fp8/bf16 |
| 自定义节点装完 ComfyUI 起不来 | 节点拉了 CUDA 依赖 | `--no-deps` 重装；逐个禁用节点定位 |
| `onnxruntime-gpu` 报 CUDA 错 | 它是 CUDA 构建 | 改装 `onnxruntime`（CPU）；人脸检测用 CPU |
| `torch.compile` 崩溃 | ROCm `reduce-overhead` 已知问题 | 不用 compile；或只试 default 模式 |
| GitHub clone 超时 | 魔搭访问 GitHub 慢 | 用 `ghfast.top` / `gitclone.com` 镜像；或本地打包上传 |
| HuggingFace 下载失败 | 网络 | 优先 ModelScope；HF 用 `HF_ENDPOINT=https://hf-mirror.com` |
| 磁盘满 | 5B+14B+缓存 | `df -h`；删 5B/缓存；模型放临时盘；只保留 fp8 |
| 中途实例关机 | 1 小时空闲 | `keepalive.py` + 保持标签页；长任务拆小；输出及时落盘 |
| 视频合成音画不同步 | TTS 时长与镜头时长不一致 | 以 TTS 时长为准调整镜头；`assemble.py` 自动按 wav 时长排版 |
| 中文字幕乱码 | 缺 CJK 字体 | 安装 `fonts-noto-cjk`；ASS 指定 `Noto Sans CJK SC` |

### 7.1 推荐的启动环境变量

```bash
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export MIOPEN_FIND_MODE=FAST
export MIOPEN_USER_DB_PATH=/mnt/workspace/ai-comic-drama/cache/miopen
export TRITON_CACHE_DIR=/mnt/workspace/ai-comic-drama/cache/triton
export HF_HOME=/mnt/workspace/ai-comic-drama/cache/huggingface
export MODELSCOPE_CACHE=/mnt/workspace/ai-comic-drama/cache/modelscope
```

不要默认开 `PYTORCH_TUNABLEOP_ENABLED=1`：第一次会非常慢，且收益不稳定；需要极限性能时再单独 A/B。

### 7.2 判断“真的在用 GPU”

```bash
rocm-smi
python - <<'PY'
import torch
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(y.dtype, y.shape, torch.cuda.max_memory_allocated()/1e9)
PY
```

不要用 `nvidia-smi` 判断；ROCm 上它要么不存在，要么是伪装。

---

## 8. 风险与回退矩阵

| 风险 | 概率 | 影响 | 预防 | 回退 |
|---|---|---|---|---|
| fp8 在 ROCm 上不可用 | 中 | 图像/视频模型跑不动 | 先 smoke；准备 bf16 | 换 bf16；降分辨率 |
| CosyVoice2 依赖装不通 | 中 | 无法配音 | 独立 venv；先测 1 句 | CPU；Edge-TTS；先做无声样片 |
| S2V 对动漫脸效果差 | 高 | 口型不可用 | 先测 1 镜头 | 不做口型；侧脸/背影/切镜 |
| 角色一致性不达标 | 高 | 观感崩 | 三视图 + 2 参考图 + 固定 Seed | 重跑；换参考图；LoRA |
| 视频闪烁/抖动 | 中 | 成片不可用 | 14B、降 CFG、4-step | 重跑；改用静帧+运镜 |
| 100GB 存储不足 | 高 | 下载中断 | fp8；分时下载；临时盘 | 删 5B/LLM/缓存 |
| 1 小时空闲关机 | 中 | 任务中断 | keepalive；拆小任务 | 持久盘续跑；日志断点 |
| 实例启动失败 | 低 | 浪费时间 | 换时段/重试 | 换实例类型 |
| GitHub 拉取失败 | 中 | 装不上 | 镜像/本地包 | 手动上传 zip |

---

## 9. 交付物与复现

每集至少保留：

```text
episodes/ep01/
├── episode.json              # 分镜与所有参数
├── keyframes/                # 关键帧
├── clips/                    # 原始片段
├── audio/                    # 对白
├── final/
│   ├── ep01.mp4
│   ├── ep01.srt
│   ├── cover.jpg
│   └── manifest.json         # 模型版本、Seed、耗时、Gate 结果
└── logs/                     # ComfyUI 日志、失败重跑记录
```

`manifest.json` 建议记录：

```json
{
  "episode": "ep01",
  "created_at": "2026-11-11T12:00:00+08:00",
  "models": {
    "image": "qwen_image_edit_2511_fp8mixed",
    "video": "wan2.2_i2v_14B_fp8_scaled + lightx2v_4steps",
    "tts": "CosyVoice2-0.5B"
  },
  "comfyui_commit": "xxxxxxx",
  "torch": "2.10.0+git...",
  "rocm": "7.2.x",
  "seeds": {"s01": 123456, "s02": 123457},
  "timings_sec": {"image_avg": 42, "video_5s_avg": 210},
  "gates": {"G0": "pass", "G1": "pass", "G6": "pass"}
}
```

复现顺序：

1. 同镜像、同 ComfyUI commit、同模型文件；
2. 同 `episode.json`、同 Seed、同工作流 JSON；
3. 先跑 1 个镜头确认环境，再批量；
4. 任何模型/工作流升级，重新跑 G3/G4。

---

## 10. 合规提醒

- **模型许可**：Qwen-Image-Edit-2511 / Wan2.2 / CosyVoice2 / Qwen3 为 Apache-2.0；Qwen-Image-2.1 为研究许可，**非商用**。
- **内容合规**：平台对 AI 生成内容有标识、审核要求；发布前按平台规则勾选/标注。
- **肖像与声音**：角色参考图、配音音色必须有授权；不要克隆真人声音用于商业内容。
- **BGM/音效/字体**：单独确认商用授权；不要使用影视原声。
- **版权**：剧本、分镜、角色设定需原创或已获授权。

---

## 附录 A：提示词模板

### A.1 全局风格前缀

```text
国漫赛璐璐风格，干净线稿，克制的冷色调，电影感构图，浅景深，9:16 竖屏，
高细节，统一的光影方向，角色设计统一，不要 3D 渲染感。
```

### A.2 角色三视图

```text
角色设定三视图：同一角色的正面、侧面、背面，站姿自然，白底，全身，
<外貌>，<服装>，<配饰>。保持三张图的脸型、发型、瞳色、服装完全一致。
国漫赛璐璐风格，干净线稿，平涂上色，无背景，无文字。
```

### A.3 关键帧编辑

```text
保持参考图 1 中角色的脸型、发型、瞳色、<标志性特征>和<服装>完全一致。
参考图 2 提供场景、光影和色调。将角色放入参考图 2 的场景，生成 9:16 竖屏构图。
镜头：<景别>，<机位>，角色位于<画面位置>。
动作/表情：<一个动作一个表情>。
风格：<全局风格前缀>。
不要改变角色身份；不要添加参考图中不存在的配饰；不要出现第二个人。
```

### A.4 视频运动

```text
镜头：<static / slow push-in / pull-out / pan left / handheld>。
动作：<一个明确动作>。
环境：<发丝/衣角/灯光/背景运动>。
保持人物面部、发型、服装与起始帧一致；保持构图稳定；动作幅度克制；
不要出现第二个人；不要快速旋转镜头。
```

### A.5 负面提示词

```text
多余手指，六指，肢体畸形，脸部崩坏，眼睛不对称，双瞳，牙齿错乱，
文字，水印，logo，签名，低清晰度，模糊，噪点，过曝，塑料皮肤，
3D 渲染感，镜头畸变，画面撕裂，闪烁，人物复制，第二个人
```

---

## 附录 B：ffmpeg 常用命令

### B.1 统一规格

```bash
ffmpeg -y -i input.mp4 \
  -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -an normalized.mp4
```

### B.2 拼接

```bash
# concat.txt 每行 file 'normalized/s01.mp4'
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy concat.mp4
```

### B.3 加字幕与混音

```bash
ffmpeg -y -i concat.mp4 -i mix.wav \
  -vf "subtitles=subs.ass:fontsdir=/usr/share/fonts" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -af "loudnorm=I=-14:TP=-1.5:LRA=11" \
  -shortest -movflags +faststart final.mp4
```

### B.4 接触表

```bash
ffmpeg -y -i clip.mp4 -vf "fps=1,scale=270:480,tile=5x3" -frames:v 1 contact.jpg
```

### B.5 音量检查

```bash
ffmpeg -i final.mp4 -af loudnorm=print_format=summary -f null -
```

---

## 附录 C：参考与出处

- 魔搭 MI300X/ROCm Notebook 环境与 SGLang 安装：<https://modelscope.cn/posts/434363>
- 魔搭免费算力与 192G 显存：<https://zhuanlan.zhihu.com/p/2073705953346720407>
- 魔搭 100GB 持久化存储：<https://github.com/modelscope/ms-cookbook/blob/main/content/chapters/chapter-04.md>
- AMD ComfyUI on Instinct 安装与基准：<https://rocm.blogs.amd.com/artificial-intelligence/comfyui/README.html>
- ROCm AI Developer Hub · Wan2.2 ComfyUI API：<https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/t2v_comfyui_api_mode_instinct.html>
- ComfyUI 官方 Wan2.2 工作流：<https://docs.comfy.org/zh/tutorials/video/wan/wan2_2>
- ComfyUI 官方 Qwen-Image-Edit-2511：<https://docs.comfy.org/zh/tutorials/image/qwen/qwen-image-edit-2511>
- Qwen-Image-2.1 模型页（研究许可）：<https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1>
- Wan2.2 repackaged 文件清单：<https://www.modelscope.cn/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged>
- Qwen-Image ComfyUI 文件：<https://www.modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI>
- CosyVoice2-0.5B：<https://www.modelscope.cn/models/iic/CosyVoice2-0.5B>
