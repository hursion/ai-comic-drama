# 开工 / 收工检查表与 Gate 记录

> 用法：每次开实例复制一份到 `episodes/ep01/logs/run-YYYYMMDD.md`，边做边勾。**不要凭记忆判断额度**。

## 1. 开工前（离线，不占 GPU）

- [ ] 故事/剧本：1 集 3–4 分钟，或 8 镜头 40 秒 teaser
- [ ] 角色卡：姓名、年龄、外貌 5 要素（脸型/发型/瞳色/服装/标志特征）
- [ ] 场景卡：3 个主场景，每个 1 段空间/光影描述
- [ ] 风格前缀：1 段全局风格（国漫/日漫/厚涂/赛璐璐）
- [ ] 负面提示词：1 份全片统一
- [ ] 音色参考：每个角色 3–10 秒干净 wav
- [ ] 分镜表：镜号、时长、景别、机位、动作、台词、情绪
- [ ] `episode.json` 初稿（参考 `episodes/ep01/episode.example.json`）
- [ ] 工作流 JSON：官方模板导出 API 格式，节点标题已按契约改名
- [ ] 本仓库已上传到 `/mnt/workspace/ai-comic-drama`
- [ ] 明确本轮要验证的 3 个问题（例如：fp8 能否出图 / 5B 速度 / S2V 中文口型）

## 2. 环境

- [ ] 实例规格：MI300X 192GB / 23 vCPU / 200GB RAM
- [ ] `nvidia-smi` 不存在是正常的；用 `rocm-smi`
- [ ] `torch.cuda.is_available()` 为 True
- [ ] `torch.version.hip` 有值
- [ ] `/mnt/workspace` 可用 > 95GB
- [ ] `ffmpeg`、`git`、`curl` 可用
- [ ] 中文字体可用（`fc-list :lang=zh` 有输出；缺则 `apt-get install -y fonts-noto-cjk`）
- [ ] 模型缓存目录已指向 `/mnt/workspace/.../cache`
- [ ] 额度页面已截图/记录（开始时长）

## 3. 安装

- [ ] `venv` 使用 `--system-site-packages`
- [ ] `pip show torch` 仍是 ROCm 版
- [ ] ComfyUI commit 已记录
- [ ] 只装了 4 个最小自定义节点
- [ ] 没有装 `xformers` / `flash-attn` / `onnxruntime-gpu`
- [ ] `curl /system_stats` 返回 MI300X
- [ ] 日志无 HIP / MIOpen 错误

## 4. 模型

- [ ] Set A 下载完成，大小校验通过
- [ ] Qwen-Image 20B T2I 能出角色三视图/场景（资产确认后可删）
- [ ] 5B 使用 `wan2.2_vae.safetensors`
- [ ] 14B 使用 `wan_2.1_vae.safetensors`
- [ ] S2V 的 `wav2vec2` 放在 `audio_encoders/`
- [ ] 文本编码器放在 `text_encoders/`
- [ ] LoRA 放在 `loras/`
- [ ] 没有同时保留 5B + 14B（或已确认放在临时盘）
- [ ] `df -h` 仍 > 15GB 余量

## 5. Gate 记录

| Gate | 目标 | 结果 | 耗时 | 备注 |
|---|---|---|---|---|
| G0 | 环境 / GPU / 磁盘 | pass / fail | | |
| G1 | ComfyUI `/system_stats` | pass / fail | | |
| G2 | 模型完整性 | pass / fail | | |
| G3 | 1 张 2K 角色图 | pass / fail | | 耗时 / 显存 |
| G4 | 1 个 5B 5 秒片段 | pass / fail | | 耗时 / 是否有噪点 |
| G5 | 三视图 + 3 场景 | pass / fail | | |
| G6 | 8 镜头 40 秒闭环 | pass / fail | | **关键 Gate** |
| G7 | 14B 单镜头对比 | pass / fail | | 5B vs 14B 20 步 vs 4 步 |
| G8 | 24–36 张关键帧 | pass / fail | | 一致性抽查 |
| G9 | 24–36 个视频片段 | pass / fail | | 重跑次数 |
| G10 | 配音 / 口型 | pass / fail | | S2V 是否可用 |
| G11 | 3–4 分钟成片 | pass / fail | | 响度 / 字幕 |

## 6. 速度基准（现场填）

| 项目 | 分辨率 / 帧数 | 步数 | 耗时 | 显存峰值 | 备注 |
|---|---|---|---|---|---|
| Qwen-Image-Edit 草稿 | 720×1280 | 4 | | | |
| Qwen-Image-Edit 成片 | 1440×2560 | 40 | | | |
| Wan2.2 5B | 1280×704 / 121f | 20 | | | |
| Wan2.2 14B 20 步 | 720p / 81f | 20 | | | |
| Wan2.2 14B 4-step | 720p / 81f | 4 | | | |
| S2V | 720p / 77f | 20 | | | |
| RIFE 2× | 5 秒片段 | — | | | |
| 合成 1 分钟 | 1080×1920 | — | | | |

> 用这组数据反推全片时间：`总时长 = 镜头数 × 单镜头耗时 × 1.3（重跑系数）`。

## 7. 收工

- [ ] 输出已写入 `/mnt/workspace/.../episodes/ep01/`
- [ ] `manifest.json` 已更新（模型、Seed、耗时、Gate）
- [ ] ComfyUI 已停止（`kill $(cat logs/comfyui.pid)`）
- [ ] Jupyter 无运行中的长任务
- [ ] 额度页面已记录结束时长
- [ ] **手动点“关闭实例”**
- [ ] 明日待办写到 `episodes/ep01/logs/todo.md`

## 8. Run Log 模板

```markdown
# Run Log · 2026-11-11

## 目标
- 验证 fp8 Qwen-Image-Edit 能否在 ROCm 7.2 出图
- 验证 Wan2.2 5B 720p 5 秒耗时

## 环境
- 镜像 / ROCm / PyTorch：
- ComfyUI commit：
- 额度开始 / 结束：

## 结果
- G3：pass，39s/张，显存峰值 41GB
- G4：pass，3m12s/片段，无噪点
- 异常：

## 决策
- 明天用 14B 4-step 跑 8 镜头

## 待办
- [ ] 补 s03 参考图
- [ ] 检查 s05 手部
```

## 9. 额度追踪

| 日期 | 开始 | 结束 | 用时 | 累计 | 主要消耗 |
|---|---|---|---|---|---|
| 2026-11-11 | | | | | 安装 + 下载 |
| 2026-11-12 | | | | | smoke + 8 镜头 |
| 2026-11-13 | | | | | 正片关键帧 |
| 2026-11-14 | | | | | 正片视频 |
| 2026-11-15 | | | | | 合成 + QC |

> 目标：累计 ≤ 75h 时完成正片，保留 ≥ 25h 给重跑。超过 80h 仍未到 G9，立即降级到 5B + 减镜头。
