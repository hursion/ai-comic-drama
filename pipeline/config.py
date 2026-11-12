"""Project paths and shared defaults."""
from __future__ import annotations

import os
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/mnt/workspace/ai-comic-drama"))
EPISODES = WORK / "episodes"
ASSETS = WORK / "assets"
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_INPUT = Path(os.environ.get("COMFY_INPUT", str(WORK / "ComfyUI" / "input")))

STYLE_PREFIX = (
    "国漫赛璐璐风格，干净线稿，克制的冷色调，电影感构图，浅景深，9:16 竖屏，"
    "高细节，统一的光影方向，角色设计统一，不要 3D 渲染感"
)

NEGATIVE_PROMPT = (
    "多余手指，六指，肢体畸形，脸部崩坏，眼睛不对称，双瞳，牙齿错乱，"
    "文字，水印，logo，签名，低清晰度，模糊，噪点，过曝，塑料皮肤，"
    "3D 渲染感，镜头畸变，画面撕裂，闪烁，人物复制，第二个人"
)
