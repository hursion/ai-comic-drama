# ComfyUI 工作流契约（API 格式）

这个目录放 **从 ComfyUI 导出的 API 格式工作流**。`pipeline/run_batch.py` 不手写工作流，
只做三件事：改提示词、换输入文件、改 seed / 步数 / 尺寸，然后 POST 到 `/prompt`。

因此每个工作流必须满足两个前提：

1. 是从 ComfyUI 菜单 `Workflow → Export (API)` 导出的（顶层是节点 id → `{class_type, inputs, _meta}`）；
2. 关键节点用下面的 **标题** 命名。脚本按标题找节点，不按节点 id。

## 1. 标题契约

| 标题 | 允许的节点类型 | 脚本写入的输入 | 哪些阶段需要 |
|---|---|---|---|
| `POSITIVE` | `CLIPTextEncode` / `TextEncodeQwenImageEditPlus` / `TextEncodeQwenImage2.1` | `text` 或 `prompt` | sheets / scenes / images / videos / flf2v |
| `NEGATIVE` | 同上 | `text` 或 `prompt` | 同上（S2V 可选） |
| `REF1` | `LoadImage` | `image` | images |
| `REF2` | `LoadImage` | `image` | images（单参考工作流可不加） |
| `START_IMAGE` | `LoadImage` | `image` | videos / flf2v / s2v |
| `END_IMAGE` | `LoadImage` | `image` | flf2v |
| `AUDIO` | `LoadAudio` | `audio` | s2v |
| `SAVE` | `SaveImage` / `VHS_VideoCombine` / `SaveVideo` | `filename_prefix` | 全部 |

补充规则：

- **标题必须唯一**。同一工作流里不要出现两个 `POSITIVE`，否则脚本会同时改两个，结果不可预期。
- 不要改节点 id；重新导出会生成新 id，脚本不依赖 id，但工作流文件要整体替换。
- `SAVE` 节点如果是 `VHS_VideoCombine`，把 `save_output` 设为 `true`，格式选 `mp4`，
  `filename_prefix` 会被脚本覆盖，不要手填镜头名。
- 视频工作流的帧数、fps、分辨率建议先在 UI 里固定；脚本只改提示词/图/seed/步数。
- 图像工作流的 `EmptyLatentImage` 尺寸可被 `--width/--height` 改；不传则用工作流默认值。

## 2. 需要准备的工作流文件

| 文件 | 用途 | 建议模板 | 必需标题 |
|---|---|---|---|
| `qwen_image_t2i.api.json` | 角色三视图、场景主视觉 | `Qwen-Image`（20B T2I） | POSITIVE / NEGATIVE / SAVE |
| `qwen_edit_2511_t2i.api.json` | 无 T2I 模型时的兜底：用空白图编辑出生图 | `Qwen-Image-Edit-2511` | POSITIVE / NEGATIVE / SAVE |
| `qwen_edit_2511_edit.api.json` | 关键帧（2 参考图编辑） | `Qwen-Image-Edit-2511` | POSITIVE / NEGATIVE / REF1 / REF2 / SAVE |
| `wan22_5b_i2v.api.json` | 5B 草稿视频 | `Wan 2.2 TI2V 5B`（I2V 用法） | POSITIVE / NEGATIVE / START_IMAGE / SAVE |
| `wan22_14b_i2v.api.json` | 14B 正片视频（可选） | `Wan 2.2 14B I2V` | POSITIVE / NEGATIVE / START_IMAGE / SAVE |
| `wan22_14b_flf2v.api.json` | 首尾帧过渡 | `Wan 2.2 14B First-Last Frame to Video` | POSITIVE / NEGATIVE / START_IMAGE / END_IMAGE / SAVE |
| `wan22_s2v.api.json` | 音频驱动口型（可选） | `Wan 2.2 S2V` | START_IMAGE / AUDIO / SAVE（POSITIVE/NEGATIVE 有就改） |

先做最小闭环只需要前 4 个：`qwen_image_t2i`、`qwen_edit_2511_edit`、`wan22_5b_i2v`，
以及可选的 `qwen_edit_2511_t2i`。

## 3. 导出步骤（每个工作流 3 分钟）

1. 浏览器打开 ComfyUI（端口映射 8188），左侧 `Workflow → Browse Templates`。
2. 选对应模板，先把模型下拉框换成 `models/comfy` 里的文件（见 [docs/MODELS.md](../docs/MODELS.md)）。
3. 在画布上重命名关键节点：右键节点 → 标题（Title）→ 输入上表中的名字。
4. 用 UI 跑一张/一段，确认节点没有报红、模型能加载。
5. `Workflow → Export (API)`，保存到本目录，文件名用上表。
6. 回到 Jupyter，运行一次标题检查：

```bash
python - <<'PY'
import json, pathlib
from pipeline.comfy_client import ComfyClient
for p in sorted(pathlib.Path("workflows").glob("*.api.json")):
    wf = json.loads(p.read_text(encoding="utf-8"))
    titles = [(t, c) for t, c, _ in ComfyClient.titles(wf) if t]
    print(p.name, "->", titles)
PY
```

7. 再跑一次 dry-run（只校验标题，不发任务；ComfyUI 必须已启动）：

```bash
python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage images --limit 1 --dry-run
```

标题不对时脚本会打印工作流里的全部标题，照着改名再导出。

## 4. 各工作流的关键参数

### 4.1 角色三视图 / 场景主视觉（T2I）

- 分辨率：`1024×1536` 或 `1080×1920`；三视图可以 `1536×1024` 横版再看情况裁。
- 步数/CFG：Qwen-Image 20B 用 30–40 步 / CFG 4.0；Lightning LoRA 用 4–8 步 / CFG 1.0。
- 提示词由 `run_batch` 自动拼：角色外貌 + 服装 + “正面/侧面/背面 + 白底 + 风格前缀”。
- 一次生成 4 张，人工挑 1 张最稳的做 `sheet`。不要直接拿第一张进批处理。
- 场景主视觉同理：空场景、无人物、统一光影，后面作为 `REF2`。

### 4.2 关键帧编辑（2511）

- 官方模板默认 2 张参考图；`REF1` 接角色三视图，`REF2` 接场景主视觉或上一镜关键帧。
- **输出分辨率跟随输入图**：在 2511 模板里把参考图后的 `ImageScale` 设成 `1080×1920`，
  或者让三视图/场景本身就是 9:16；否则出图是 3:2/4:3，后期只能加黑边。
  `run_batch --width/--height` 只改 T2I 的 `Empty*Latent*`，不改编辑输出。
- 草稿：Lightning LoRA，4 步 / CFG 1.0 / 720×1280。
- 正片：40 步 / CFG 4.0 / 1080×1920（或 1440×2560 再缩）。
- 如果只有 1 张参考图，把 `REF2` 节点从工作流里删掉，`run_batch` 会自动跳过（`required=False`）。
- 2511 的 `TextEncodeQwenImageEditPlus` 可能要求图像输入；若你的 T2I 工作流跑不通，
  说明它不能无图生成，改用 `qwen_image_t2i`（20B T2I）。

### 4.3 Wan2.2 5B 草稿

- 帧数 `121`（4n+1），24fps ≈ 5 秒；分辨率 `1280×704` 或竖屏 `704×1280`。
- 20 步 / CFG 5.0 起步；先跑 2 个镜头看运动是否成立。
- 5B 是“镜头语言验证器”，不要拿它做最终画质。

### 4.4 Wan2.2 14B I2V / FLF2V

- 高噪 + 低噪两个 diffusion 节点，各自挂对应的 LightX2V 4-step LoRA。
- 4 步 / CFG 1.0 先跑；不够再 8 步，最后才 20 步 / CFG 3.5。
- 81 帧 / 16fps ≈ 5 秒；输出后统一用 ffmpeg 补到 30fps。
- FLF2V 必须同时有 `START_IMAGE` 和 `END_IMAGE`；缺一张脚本会跳过该镜头。

### 4.5 Wan2.2 S2V

- 77 帧 / 16fps 一段；长对白要分段生成再拼。
- 输入是 `START_IMAGE` + `AUDIO`（wav，建议 16k/24k 单声道）。
- 动漫脸口型容易漂移，先拿 1 个中近景镜头测，不要整集直接上。

### 4.6 cuda24g / GGUF 工作流（24GB 显存）

- 安装 `ComfyUI-GGUF`（`INSTALL_GGUF=1` 时 bootstrap 会自动装）。
- GGUF diffusion 模型放在 `models/comfy/unet/`，不是 `diffusion_models/`。
- 在官方 Qwen-Image / Qwen-Image-Edit-2511 模板里，把 `UNETLoader` / `Load Diffusion Model`
  换成 **`Unet Loader (GGUF)`**，选择 `qwen-image-edit-2511-Q4_K_M.gguf` 或 `qwen-image-Q4_K_M.gguf`。
- 文本编码器仍用原生 fp8：`qwen_2.5_vl_7b_fp8_scaled.safetensors`；VAE 仍用 `qwen_image_vae.safetensors`。
  这样避免 GGUF CLIP / mmproj 的额外兼容问题。
- **第一次 smoke 不要挂 Lightning 4-step LoRA**。GGUF + LoRA 需要单独验证；先用 20 步 / CFG 4.0。
- 图像输出建议 720×1280 草稿，1080×1920 正片；VAE 解码用 `VAEDecodeTiled` 降低显存峰值。
- 视频仍用 Wan2.2 5B fp16 + `umt5_xxl_fp8_e4m3fn_scaled.safetensors`；
  分辨率 480×832、81 帧、16fps 起步，不要直接 720p 121 帧。
- 导出后的文件名可以不变（`qwen_edit_2511_edit.api.json` 等），脚本只按节点标题改输入。

## 5. 输出文件约定

`ComfyClient.download_outputs` 会遍历 `outputs` 里的 `images / gifs / videos / audio / files`，
把第一个输出重命名为目标文件（如 `keyframes/s01_start.png`）。因此：

- `SAVE` 节点必须真正把文件写进 ComfyUI 的 `output` 目录（`save_output=true`）；
- 一个工作流只保留一个 `SAVE` 节点，避免多文件抢同一个目标名；
- 视频用 `VHS_VideoCombine` 时，输出项在 `gifs` 里是正常的，不要以为是 GIF。

## 6. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `missing node titles [...]` | 节点没改名，或导出的是 UI 格式 | 按第 3 步改名，`Export (API)` 重导 |
| `queue rejected: ... node_errors` | 模型文件名/路径不对，或节点输入类型不匹配 | 看返回的 node_errors，回 UI 重选模型 |
| `workflow finished but produced no downloadable output` | `SAVE` 没开保存，或输出在临时目录 | 打开 `save_output`，确认 `output` 目录有文件 |
| 下载下来是 0 字节 | `/view` 参数里的 `type/subfolder` 不对 | 用 UI 的“打开输出目录”核对 |
| 视频只有 1 帧 | 帧数不是 4n+1，或 VAE 不匹配 | 改 121/81 帧；5B 用 `wan2.2_vae`，14B 用 `wan_2.1_vae` |
| 颜色噪声/花屏 | 用了错误的 VAE | 14B 必须 `wan_2.1_vae.safetensors` |
