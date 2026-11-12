"""Generate a structured storyboard (episode.json) with a local Qwen3 model.

This is intentionally a one-shot generator: feed it a short story + style guide,
ask for strict JSON, validate it, write it. If the JSON is truncated, lower
--shots or split the story into acts and merge manually.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SYSTEM = """你是一名竖屏 AI 漫剧的分镜导演。你只输出严格 JSON，不输出解释、Markdown 或代码围栏。
你必须保证：
1. 每个镜头 4-6 秒，镜头之间动作连续、空间关系清楚。
2. 每个镜头只包含一个主要动作和一个机位。
3. 所有画面都是 9:16 竖屏，风格统一。
4. image_prompt 只描述这一帧的静态画面；video_prompt 只描述运动，不重复外貌。
5. 角色外貌在每个镜头中保持一致，不新增参考图中不存在的配饰。
"""

SCHEMA_HINT = """输出 JSON 结构：
{
  "title": "剧名",
  "style": "全局风格前缀",
  "aspect_ratio": "9:16",
  "fps": 30,
  "characters": [
    {"id": "lin", "name": "林晚", "appearance": "外貌 5 要素", "outfit": "服装", "sheet": "assets/characters/lin_sheet.png", "voice_ref": "assets/voices/lin.wav"}
  ],
  "locations": [
    {"id": "elevator", "name": "电梯", "master": "assets/scenes/elevator_master.png", "prompt": "空间与光影"}
  ],
  "shots": [
    {
      "id": "s01",
      "duration": 5,
      "location": "elevator",
      "characters": ["lin"],
      "shot_size": "medium",
      "camera": "slow push-in",
      "action": "一个动作",
      "dialogue": [{"character": "lin", "text": "台词", "emotion": "情绪"}],
      "image_prompt": "静态画面描述",
      "video_prompt": "运动描述",
      "negative_prompt": "负面提示词",
      "refs": ["assets/characters/lin_sheet.png", "assets/scenes/elevator_master.png"],
      "keyframe_start": "keyframes/s01_start.png",
      "clip": "clips/s01.mp4"
    }
  ]
}
"""


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON (likely truncated); lower --shots or --max-new-tokens")


def validate(ep: dict) -> None:
    for key in ("title", "characters", "locations", "shots"):
        if key not in ep:
            raise ValueError(f"missing key: {key}")
    if not ep["shots"]:
        raise ValueError("no shots")
    ids = set()
    for i, shot in enumerate(ep["shots"]):
        shot.setdefault("id", f"s{i+1:02d}")
        if shot["id"] in ids:
            raise ValueError(f"duplicate shot id {shot['id']}")
        ids.add(shot["id"])
        shot.setdefault("duration", 5)
        shot.setdefault("keyframe_start", f"keyframes/{shot['id']}_start.png")
        shot.setdefault("clip", f"clips/{shot['id']}.mp4")
        shot.setdefault("refs", [])
        shot.setdefault("dialogue", [])
        shot.setdefault("enabled", True)
    ep.setdefault("style", "国漫赛璐璐，干净线稿，电影感构图，9:16 竖屏")
    ep.setdefault("aspect_ratio", "9:16")
    ep.setdefault("fps", 30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="models/qwen3-8b")
    ap.add_argument("--shots", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    story = Path(args.story).read_text(encoding="utf-8")
    user = (
        f"请把下面的故事改编成 {args.shots} 个镜头的竖屏 AI 漫剧分镜。\n"
        f"只输出 JSON，不要解释。\n\n{SCHEMA_HINT}\n\n故事：\n{story}"
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    torch.manual_seed(args.seed)

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
            repetition_penalty=1.05,
        )
    response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    try:
        ep = extract_json(response)
    except Exception as e:  # noqa: BLE001
        raw = Path(args.out).with_suffix(".raw.txt")
        raw.write_text(response, encoding="utf-8")
        print(f"parse failed: {e}; raw output saved to {raw}")
        return 1

    validate(ep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}: {len(ep['shots'])} shots, {len(ep['characters'])} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
