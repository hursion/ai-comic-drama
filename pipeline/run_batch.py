"""Batch-run ComfyUI API workflows for an episode.

The episode JSON is the single source of truth. Workflows must be exported in
ComfyUI API format and use the node titles documented in workflows/README.md.

Examples:
  python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage sheets
  python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage images --limit 2
  python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage videos --limit 2
  python -m pipeline.run_batch --episode episodes/ep01/episode.json --stage s2v --limit 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import wave
from pathlib import Path

from pipeline import config
from pipeline.comfy_client import ComfyClient, ComfyError

DEFAULT_WORKFLOWS = {
    "sheets": "workflows/qwen_image_t2i.api.json",
    "images": "workflows/qwen_edit_2511_edit.api.json",
    "videos": "workflows/wan22_5b_i2v.api.json",
    "flf2v": "workflows/wan22_14b_flf2v.api.json",
    "s2v": "workflows/wan22_s2v.api.json",
}


def stable_seed(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


EPISODE_SUBDIRS = {"keyframes", "clips", "audio", "final", "normalized"}


def resolve(path_str: str, episode_dir: Path) -> Path:
    """Resolve a path from episode.json.

    - assets/..., models/..., workflows/... are project-relative (WORK)
    - keyframes/..., clips/..., audio/... are episode-relative
    - absolute paths win
    """
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in EPISODE_SUBDIRS:
        return (episode_dir / p).resolve()
    project = (config.WORK / p).resolve()
    if project.exists():
        return project
    local = (episode_dir / p).resolve()
    if local.exists():
        return local
    return project


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_titles(client_titles: list[tuple[str, str, str]], required: list[str]) -> None:
    have = {t for t, _c, _i in client_titles if t}
    missing = [t for t in required if t not in have]
    if missing:
        print("workflow titles:")
        for t, c, i in client_titles:
            print(f"  {i:>6}  {c:<45} {t}")
        raise ComfyError(
            f"missing node titles {missing}. Rename the nodes in ComfyUI and re-export API JSON."
        )


def shot_audio(shot: dict, episode_dir: Path) -> Path:
    """Return one wav for S2V.

    TTS writes audio/<shot_id>_<line>.wav. S2V takes a single audio file, so
    concatenate the line wavs into audio/<shot_id>.wav on first use.
    """
    explicit = shot.get("audio")
    if explicit:
        return resolve(explicit, episode_dir)
    merged = episode_dir / "audio" / f"{shot['id']}.wav"
    if merged.is_file():
        return merged
    parts = list((episode_dir / "audio").glob(f"{shot['id']}_*.wav"))
    parts.sort(key=lambda p: int(p.stem.rsplit("_", 1)[-1]) if p.stem.rsplit("_", 1)[-1].isdigit() else 0)
    if not parts:
        return merged
    with wave.open(str(parts[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
    for p in parts[1:]:
        with wave.open(str(p), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (
                    params.nchannels, params.sampwidth, params.framerate):
                raise ComfyError(f"{p} has different wav params; keep CosyVoice output consistent")
            frames.append(w.readframes(w.getnframes()))
    merged.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(merged), "wb") as out:
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)
    return merged


def run_one(client: ComfyClient, workflow: dict, out_path: Path, prefix: str,
            timeout: float, force: bool, dry_run: bool) -> Path | None:
    if out_path.exists() and not force:
        print(f"  skip (exists): {out_path}")
        return out_path
    client.set_prefix(workflow, prefix)
    if dry_run:
        print(f"  dry-run -> {out_path}")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    outs = client.run(workflow, out_dir=out_path.parent, prefix=prefix, timeout=timeout)
    if not outs:
        raise ComfyError("workflow finished but produced no downloadable output")
    produced = outs[0]
    if produced != out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        produced.replace(out_path)
    dt = time.perf_counter() - t0
    print(f"  ok {out_path}  ({dt:.1f}s)")
    return out_path


def stage_sheets(client: ComfyClient, episode: dict, episode_dir: Path,
                 args, wf: dict) -> int:
    """Generate character turnaround sheets and location master images."""
    require_titles(ComfyClient.titles(wf), ["POSITIVE", "NEGATIVE", "SAVE"])
    failures = 0
    jobs: list[tuple[str, str, Path]] = []

    for ch in episode.get("characters", []):
        out = resolve(ch.get("sheet", f"assets/characters/{ch['id']}_sheet.png"), episode_dir)
        prompt = (
            f"角色设定三视图：同一角色的正面、侧面、背面，站姿自然，白底，全身。\n"
            f"角色：{ch.get('name', ch['id'])}。{ch.get('appearance', '')}。"
            f"服装：{ch.get('outfit', '')}。\n"
            f"保持三张图的脸型、发型、瞳色、服装完全一致。\n"
            f"风格：{config.STYLE_PREFIX}。无背景，无文字。"
        )
        jobs.append((ch["id"], prompt, out))

    for loc in episode.get("locations", []):
        out = resolve(loc.get("master", f"assets/scenes/{loc['id']}_master.png"), episode_dir)
        prompt = (
            f"场景主视觉：{loc.get('name', loc['id'])}。{loc.get('prompt', '')}。\n"
            f"空场景，无人物，无文字，统一光影，9:16 竖屏，电影感构图。\n"
            f"风格：{config.STYLE_PREFIX}。"
        )
        jobs.append((f"loc_{loc['id']}", prompt, out))

    for job_id, prompt, out in jobs:
        out.parent.mkdir(parents=True, exist_ok=True)
        wf2 = json.loads(json.dumps(wf))
        ComfyClient.set_prompts(wf2, prompt, config.NEGATIVE_PROMPT)
        ComfyClient.set_seed(wf2, stable_seed(episode["title"], job_id, "sheet"))
        if args.steps is not None or args.cfg is not None:
            ComfyClient.set_steps_cfg(wf2, args.steps, args.cfg)
        if args.width and args.height:
            ComfyClient.set_size(wf2, args.width, args.height)
        print(f"[asset] {job_id} -> {out}")
        try:
            run_one(client, wf2, out, f"{episode.get('slug', 'ep')}_{job_id}_asset",
                    args.timeout, args.force, args.dry_run)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: {e}")
    return failures


def stage_images(client: ComfyClient, episode: dict, episode_dir: Path, workflow: dict,
                 args) -> int:
    require_titles(ComfyClient.titles(workflow), ["POSITIVE", "NEGATIVE", "SAVE", "REF1"])
    failures = 0
    shots = [s for s in episode.get("shots", []) if s.get("enabled", True)]
    for shot in shots[: args.limit or None]:
        out = resolve(shot.get("keyframe_start", f"keyframes/{shot['id']}_start.png"), episode_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        refs = shot.get("refs") or []
        ref_paths = [resolve(r, episode_dir) for r in refs]
        ref_paths = [p for p in ref_paths if p.is_file()]
        if not ref_paths:
            print(f"[image] {shot['id']}: no existing refs, skipped (run --stage sheets first)")
            failures += 1
            continue
        wf = json.loads(json.dumps(workflow))
        try:
            names = [client.stage_file(p, prefix=f"{shot['id']}_") for p in ref_paths[:2]]
            ComfyClient.patch(wf, "REF1", "image", names[0])
            if len(names) > 1:
                ComfyClient.patch(wf, "REF2", "image", names[1], required=False)
            prompt = f"全局风格：{config.STYLE_PREFIX}\n\n{shot.get('image_prompt', '')}"
            ComfyClient.set_prompts(wf, prompt, shot.get("negative_prompt") or config.NEGATIVE_PROMPT)
            seed = int(shot.get("seed") or stable_seed(episode["title"], shot["id"], "image"))
            ComfyClient.set_seed(wf, seed)
            if args.steps is not None or args.cfg is not None:
                ComfyClient.set_steps_cfg(wf, args.steps, args.cfg)
            if args.width and args.height:
                ComfyClient.set_size(wf, args.width, args.height)
            print(f"[image] {shot['id']} -> {out}  seed={seed}")
            run_one(client, wf, out, f"{episode.get('slug', 'ep')}_{shot['id']}_img",
                    args.timeout, args.force, args.dry_run)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: {e}")
    return failures


def stage_videos(client: ComfyClient, episode: dict, episode_dir: Path, workflow: dict,
                 args, flf: bool = False) -> int:
    required = ["POSITIVE", "NEGATIVE", "SAVE", "START_IMAGE"]
    if flf:
        required.append("END_IMAGE")
    require_titles(ComfyClient.titles(workflow), required)
    failures = 0
    shots = [s for s in episode.get("shots", []) if s.get("enabled", True)]
    for shot in shots[: args.limit or None]:
        out = resolve(shot.get("clip", f"clips/{shot['id']}.mp4"), episode_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        start = resolve(shot.get("keyframe_start", f"keyframes/{shot['id']}_start.png"), episode_dir)
        if not start.exists():
            print(f"[video] {shot['id']}: missing start keyframe {start}, skipped")
            failures += 1
            continue
        wf = json.loads(json.dumps(workflow))
        try:
            name = client.stage_file(start, prefix=f"{shot['id']}_start_")
            ComfyClient.patch(wf, "START_IMAGE", "image", name)
            if flf:
                end = resolve(shot.get("keyframe_end", ""), episode_dir)
                if not end.is_file():
                    print(f"[video] {shot['id']}: missing end keyframe, skipped")
                    failures += 1
                    continue
                ComfyClient.patch(wf, "END_IMAGE", "image", client.stage_file(end, prefix=f"{shot['id']}_end_"))
            prompt = f"全局风格：{config.STYLE_PREFIX}\n\n{shot.get('video_prompt', '')}"
            ComfyClient.set_prompts(wf, prompt, shot.get("negative_prompt") or config.NEGATIVE_PROMPT)
            seed = int(shot.get("seed") or stable_seed(episode["title"], shot["id"], "video"))
            ComfyClient.set_seed(wf, seed)
            if args.steps is not None or args.cfg is not None:
                ComfyClient.set_steps_cfg(wf, args.steps, args.cfg)
            if args.width and args.height:
                ComfyClient.set_size(wf, args.width, args.height)
            print(f"[video] {shot['id']} -> {out}  seed={seed}")
            run_one(client, wf, out, f"{episode.get('slug', 'ep')}_{shot['id']}_vid",
                    args.timeout, args.force, args.dry_run)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: {e}")
    return failures


def stage_s2v(client: ComfyClient, episode: dict, episode_dir: Path, workflow: dict,
              args) -> int:
    require_titles(ComfyClient.titles(workflow), ["SAVE", "START_IMAGE", "AUDIO"])
    failures = 0
    shots = [s for s in episode.get("shots", []) if s.get("enabled", True)]
    for shot in shots[: args.limit or None]:
        audio_p = shot_audio(shot, episode_dir)
        if not audio_p.is_file():
            print(f"[s2v] {shot['id']}: missing audio {audio_p}, skipped")
            failures += 1
            continue
        start = resolve(shot.get("keyframe_start", f"keyframes/{shot['id']}_start.png"), episode_dir)
        if not start.is_file():
            print(f"[s2v] {shot['id']}: missing keyframe {start}, skipped")
            failures += 1
            continue
        out = resolve(shot.get("s2v_clip", f"clips/{shot['id']}_s2v.mp4"), episode_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        wf = json.loads(json.dumps(workflow))
        try:
            ComfyClient.patch(wf, "START_IMAGE", "image",
                              client.stage_file(start, prefix=f"{shot['id']}_s2v_"))
            ComfyClient.patch_any(wf, "AUDIO", ["audio"], client.stage_file(audio_p, prefix=f"{shot['id']}_"))
            ComfyClient.patch_any(wf, "POSITIVE", ["text", "prompt"],
                                  f"角色说话，口型自然，保持面部稳定。{shot.get('video_prompt', '')}",
                                  required=False)
            ComfyClient.patch_any(wf, "NEGATIVE", ["text", "prompt"], config.NEGATIVE_PROMPT,
                                  required=False)
            ComfyClient.set_seed(wf, stable_seed(episode["title"], shot["id"], "s2v"))
            print(f"[s2v] {shot['id']} -> {out}")
            run_one(client, wf, out, f"{episode.get('slug', 'ep')}_{shot['id']}_s2v",
                    args.timeout, args.force, args.dry_run)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: {e}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--stage", required=True, choices=["sheets", "images", "videos", "flf2v", "s2v"])
    ap.add_argument("--workflow", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=7200)
    args = ap.parse_args()

    image_stage = args.stage in ("sheets", "images")
    if args.steps is None:
        v = os.environ.get("IMAGE_STEPS" if image_stage else "VIDEO_STEPS")
        if v:
            args.steps = int(v)
    if args.cfg is None:
        v = os.environ.get("IMAGE_CFG" if image_stage else "VIDEO_CFG")
        if v:
            args.cfg = float(v)
    if args.width is None:
        v = os.environ.get("IMAGE_WIDTH" if image_stage else "VIDEO_WIDTH")
        if v:
            args.width = int(v)
    if args.height is None:
        v = os.environ.get("IMAGE_HEIGHT" if image_stage else "VIDEO_HEIGHT")
        if v:
            args.height = int(v)

    ep_path = Path(args.episode)
    episode = load_json(ep_path)
    episode_dir = ep_path.parent
    episode.setdefault("slug", ep_path.stem)
    wf_path = Path(args.workflow or DEFAULT_WORKFLOWS[args.stage])
    if not wf_path.is_absolute():
        wf_path = config.WORK / wf_path
    if not wf_path.is_file():
        print(f"workflow not found: {wf_path}")
        print("Export it from ComfyUI (Workflow -> Export (API)) and rename nodes, see workflows/README.md")
        return 2
    workflow = load_json(wf_path)

    client = ComfyClient(config.COMFY_URL, input_dir=config.COMFY_INPUT)
    stats = client.system_stats()
    print("device:", stats.get("devices", [{}])[0].get("name", "unknown"))
    print("workflow:", wf_path)
    print("episode :", ep_path)

    t0 = time.perf_counter()
    if args.stage == "sheets":
        failures = stage_sheets(client, episode, episode_dir, args, workflow)
    elif args.stage == "images":
        failures = stage_images(client, episode, episode_dir, workflow, args)
    elif args.stage == "videos":
        failures = stage_videos(client, episode, episode_dir, workflow, args)
    elif args.stage == "flf2v":
        failures = stage_videos(client, episode, episode_dir, workflow, args, flf=True)
    elif args.stage == "s2v":
        failures = stage_s2v(client, episode, episode_dir, workflow, args)
    else:
        raise SystemExit("unknown stage")

    print(f"\nstage={args.stage} failures={failures} elapsed={time.perf_counter()-t0:.1f}s")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
