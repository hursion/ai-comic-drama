# assets 目录约定

这个目录放“跨镜头复用”的素材。所有路径都相对于项目根目录
`/mnt/workspace/ai-comic-drama`，`episode.json` 里直接写 `assets/...`。

```text
assets/
  characters/
    lin_sheet.png        # 角色三视图，正面/侧面/背面，白底
  scenes/
    elevator_master.png  # 场景主视觉，空场景，无人物
    corridor_master.png
  voices/
    lin.wav              # 3-10 秒干净人声，无 BGM、无混响
  bgm/
    theme.mp3            # 可选，最终混音用
```

## 角色三视图（必须）

- 由 `python -m pipeline.run_batch --stage sheets` 生成，或手动在 ComfyUI 里生成后挑一张。
- 要求：同一角色的正面、侧面、背面在同一张图里；白底或浅灰底；全身；服装一致。
- 角色脸型、发型、瞳色、服装、配饰五要素必须一眼能认出来。模糊、遮挡、半身都不合格。
- 文件建议 1024×1536 或 1536×1024，PNG。

## 场景主视觉（必须）

- 空场景、无人物、无文字，光影方向固定。
- 后面作为 Qwen-Image-Edit 的 `REF2`，所以不要出现会跟角色冲突的主体。
- 一个场景一张，不要用电影截图（版权 + 风格不可控）。

## 参考音色（CosyVoice2 用）

- 3-10 秒，单人，干净，无背景音乐、无混响、无笑声。
- 16k 或 24k wav，单声道即可。
- `episode.json` 里的 `voice_text` 必须是这段音频的**准确文字**，否则 zero-shot 音色会漂。
- 没有参考音时，可以临时用 `--sft-speaker 中文女` 让 CosyVoice2 走内置音色；
  这只是为了跑通链路，正式配音仍建议用自己的参考音。

## 版权

- 角色、场景、参考音必须是自有或已获授权。
- 不要用明星脸、影视截图、他人画作做参考图。
- 生成结果是否可商用取决于模型许可证，见 [docs/MODELS.md](../docs/MODELS.md)。
