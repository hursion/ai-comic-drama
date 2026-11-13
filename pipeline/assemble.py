"""Assemble normalized clips + dialogue + optional BGM + ASS subtitles into one MP4.

This deliberately stays on ffmpeg instead of moviepy: predictable, fast, and
easy to debug on a headless ROCm box.

Examples:
  python -m pipeline.assemble --episode episodes/ep01/episode.json --draft
  python -m pipeline.assemble --episode episodes/ep01/episode.json --bgm assets/bgm/theme.mp3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from pipeline import config


def run(cmd: list[str], dry: bool = False) -> None:
    print("+", " ".join(cmd))
    if dry:
        return
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ], text=True).strip()
    return float(out)


def make_silence(path: Path, seconds: float, rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = int(seconds * rate)
        w.writeframes(b"\x00\x00\x00\x00" * frames)


def normalize_clip(src: Path, dst: Path, width: int, height: int, fps: int,
                   duration: float, preset: str, crf: int, dry: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, float(duration))
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={duration:.3f}"
    )
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf, "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an", str(dst),
    ], dry=dry)


def dialogue_files(shot: dict, episode_dir: Path) -> list[Path]:
    dialogue = shot.get("dialogue") or []
    if not dialogue:
        return []
    single = shot.get("audio")
    if single:
        p = Path(single)
        return [p if p.is_absolute() else episode_dir / p]
    out = []
    for i in range(len(dialogue)):
        p = episode_dir / "audio" / f"{shot['id']}_{i}.wav"
        if not p.exists():
            alt = episode_dir / "audio" / f"{shot['id']}.wav"
            p = alt if alt.exists() else p
        out.append(p)
    return out


def write_ass(episode: dict, episode_dir: Path, shots: list[dict], out: Path) -> Path:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Noto Sans CJK SC,58,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,3,1,2,70,70,150,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    cursor = 0.0
    for shot in shots:
        dur = float(shot.get("duration", 5))
        files = dialogue_files(shot, episode_dir)
        dialogue = shot.get("dialogue") or []
        inner = 0.25
        for i, line in enumerate(dialogue):
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            tts_dur = 2.5
            if i < len(files) and files[i].exists():
                try:
                    tts_dur = ffprobe_duration(files[i])
                except Exception:
                    pass
            start = cursor + float(line.get("start", inner))
            end = min(cursor + dur - 0.05, start + max(1.0, tts_dur))
            lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{ass_escape(text)}"
            )
            inner = (start - cursor) + tts_dur + 0.15
        cursor += dur
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def write_srt(episode: dict, episode_dir: Path, shots: list[dict], out: Path) -> Path:
    lines = []
    idx = 0
    cursor = 0.0
    for shot in shots:
        dur = float(shot.get("duration", 5))
        files = dialogue_files(shot, episode_dir)
        dialogue = shot.get("dialogue") or []
        inner = 0.25
        for i, line in enumerate(dialogue):
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            tts_dur = 2.5
            if i < len(files) and files[i].exists():
                try:
                    tts_dur = ffprobe_duration(files[i])
                except Exception:
                    pass
            start = cursor + float(line.get("start", inner))
            end = min(cursor + dur - 0.05, start + max(1.0, tts_dur))
            idx += 1
            lines += [str(idx), f"{srt_time(start)} --> {srt_time(end)}", text, ""]
            inner = (start - cursor) + tts_dur + 0.15
        cursor += dur
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--bgm", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--no-subtitles", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg/ffprobe not found. Install: apt-get install -y ffmpeg (or use imageio-ffmpeg)")
        return 2

    ep_path = Path(args.episode)
    episode = json.loads(ep_path.read_text(encoding="utf-8"))
    episode_dir = ep_path.parent
    width, height = (720, 1280) if args.draft else (1080, 1920)
    crf = args.crf if args.crf is not None else (23 if args.draft else 18)
    preset = "veryfast" if args.draft else "medium"
    out_dir = episode_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(args.out) if args.out else out_dir / f"{ep_path.parent.name}.mp4"

    shots = [s for s in episode.get("shots", []) if s.get("enabled", True)]
    if not shots:
        print("no enabled shots")
        return 2

    norm_dir = episode_dir / "normalized"
    norm_files = []
    playable: list[dict] = []
    print("== normalize ==")
    for shot in shots:
        clip = Path(shot.get("clip", f"clips/{shot['id']}.mp4"))
        if not clip.is_absolute():
            clip = episode_dir / clip
        if not clip.exists():
            print(f"WARN: missing clip {clip}, skip")
            continue
        dst = norm_dir / f"{shot['id']}.mp4"
        normalize_clip(clip, dst, width, height, args.fps, float(shot.get("duration", 5)),
                       preset, crf, args.dry_run)
        norm_files.append(dst)
        playable.append(shot)

    if not norm_files:
        print("no clips to assemble")
        return 2
    total = sum(float(s.get("duration", 5)) for s in playable)

    concat_txt = norm_dir / "concat.txt"
    concat_txt.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in norm_files) + "\n", encoding="utf-8"
    )
    concat_mp4 = norm_dir / "concat.mp4"
    print("== concat ==")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
         "-c", "copy", str(concat_mp4)], dry=args.dry_run)

    print("== audio ==")
    silence = norm_dir / "silence.wav"
    if not args.dry_run:
        make_silence(silence, max(0.1, total))
    mix = norm_dir / "mix.wav"
    inputs = ["-i", str(silence)]
    filters = []
    labels = ["[0:a]"]
    idx = 1
    cursor = 0.0
    for shot in playable:
        dur = float(shot.get("duration", 5))
        files = dialogue_files(shot, episode_dir)
        dialogue = shot.get("dialogue") or []
        inner = 0.25
        for i, line in enumerate(dialogue):
            if i >= len(files) or not files[i].exists():
                inner += 2.5
                continue
            inputs += ["-i", str(files[i])]
            delay_ms = int(round((cursor + float(line.get("start", inner))) * 1000))
            filters.append(
                f"[{idx}:a]adelay=delays={delay_ms}:all=1,"
                f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"volume=1.0[a{idx}]"
            )
            labels.append(f"[a{idx}]")
            try:
                inner = (float(line.get("start", inner))) + ffprobe_duration(files[i]) + 0.15
            except Exception:
                inner += 2.7
            idx += 1
        cursor += dur
    if idx > 1:
        filters.append(
            "".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=longest[aout]"
        )
        run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
             "-map", "[aout]", "-c:a", "pcm_s16le", str(mix)], dry=args.dry_run)
    else:
        if not args.dry_run:
            shutil.copy2(silence, mix)
        print("WARN: no dialogue audio found; using silence")

    print("== subtitles ==")
    ass = write_ass(episode, episode_dir, playable, out_dir / f"{ep_path.parent.name}.ass")
    srt = write_srt(episode, episode_dir, playable, out_dir / f"{ep_path.parent.name}.srt")
    print("wrote", ass, srt)

    print("== final mux ==")
    vf = None
    if not args.no_subtitles:
        vf = f"subtitles={ass.as_posix()}:fontsdir=/usr/share/fonts"
    cmd = ["ffmpeg", "-y", "-i", str(concat_mp4)]
    af = None
    if args.bgm:
        bgm = Path(args.bgm)
        if not bgm.is_absolute():
            bgm = config.WORK / bgm
        if bgm.exists():
            cmd += ["-stream_loop", "-1", "-i", str(bgm)]
            # voice [1:a], bgm [2:a]; loudnorm must live in the same filtergraph
            fc = (
                "[1:a]volume=1.0[voice];[2:a]volume=0.12[bg];"
                "[voice][bg]amix=inputs=2:duration=first:dropout_transition=2[mixed];"
                "[mixed]loudnorm=I=-14:TP=-1.5:LRA=11,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[aout]"
            )
            cmd += ["-filter_complex", fc, "-map", "0:v:0", "-map", "[aout]"]
        else:
            print(f"WARN: bgm not found: {bgm}")
            cmd += ["-i", str(mix), "-map", "0:v:0", "-map", "1:a:0"]
            af = "loudnorm=I=-14:TP=-1.5:LRA=11,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    else:
        cmd += ["-i", str(mix), "-map", "0:v:0", "-map", "1:a:0"]
        af = "loudnorm=I=-14:TP=-1.5:LRA=11,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    if vf:
        cmd += ["-vf", vf]
    cmd += [
        "-c:v", "libx264", "-preset", "slow" if not args.draft else "veryfast",
        "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
    ]
    if af:
        cmd += ["-af", af]
    cmd += ["-shortest", "-movflags", "+faststart", str(out_mp4)]
    try:
        run(cmd, dry=args.dry_run)
    except subprocess.CalledProcessError:
        if vf:
            print("subtitle burn-in failed; retrying without subtitles")
            cmd = [c for c in cmd if c != vf and c != "-vf"]
            run(cmd, dry=args.dry_run)
        else:
            raise

    if not args.dry_run and out_mp4.exists():
        cover = out_dir / "cover.jpg"
        for ss in ("1", "0"):
            try:
                subprocess.run(["ffmpeg", "-y", "-ss", ss, "-i", str(out_mp4), "-frames:v", "1",
                                "-q:v", "2", str(cover)], check=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"cover at {ss}s failed:", e)

    manifest = {
        "episode": episode.get("title", ep_path.parent.name),
        "output": str(out_mp4),
        "width": width,
        "height": height,
        "fps": args.fps,
        "duration_sec": round(total, 2),
        "shots": [s["id"] for s in playable],
        "clips": [str(p) for p in norm_files],
        "subtitles": {"ass": str(ass), "srt": str(srt)},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("done:", out_mp4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
