#!/usr/bin/env python3
"""Download ComfyUI / TTS / LLM weights for the AI comic drama pipeline.

Design goals:
- only download the files we actually need (the Wan2.2 repackaged repo is huge)
- verify sizes so a truncated download fails loudly
- keep real bytes in one cache dir and expose stable paths via symlinks
- make it obvious what will fit in the 100GB persistent disk

Examples:
  python scripts/02_download_models.py --list
  python scripts/02_download_models.py --groups image,video-5b --dry-run
  python scripts/02_download_models.py --groups image,video-5b,tts,llm,asr
  python scripts/02_download_models.py --groups video-14b
  python scripts/02_download_models.py --verify
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/mnt/workspace/ai-comic-drama"))
DEST = WORK / "models"
CACHE = WORK / "cache" / "downloads"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
os.environ.setdefault("MODELSCOPE_CACHE", str(WORK / "cache" / "modelscope"))

# group -> list of entries
# kind: ms_file | ms_snapshot | hf_file | url
MANIFEST = [
    # ---- image: Qwen-Image-Edit-2511 fp8 ----
    {"groups": ["image"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
     "remote": "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
     "dest": "comfy/diffusion_models", "gb": 20.5},
    {"groups": ["image"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "dest": "comfy/text_encoders", "gb": 9.38},
    {"groups": ["image"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/vae/qwen_image_vae.safetensors",
     "dest": "comfy/vae", "gb": 0.25},

    # ---- reliable text-to-image for character sheets / scene masters ----
    # Only needed once per project. After the assets are approved you can delete
    # this 20.4GB file to make room for final renders; see docs/MODELS.md.
    {"groups": ["image-t2i"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
     "dest": "comfy/diffusion_models", "gb": 20.4},
    {"groups": ["image-t2i"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "dest": "comfy/text_encoders", "gb": 9.38},
    {"groups": ["image-t2i"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/vae/qwen_image_vae.safetensors",
     "dest": "comfy/vae", "gb": 0.25},

    # ---- optional image bf16 fallback ----
    {"groups": ["image-bf16"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
     "remote": "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
     "dest": "comfy/diffusion_models", "gb": 40.8},

    # ---- optional Qwen-Image-Edit-2511 Lightning 4-step LoRA ----
    {"groups": ["image-lora"], "kind": "ms_file",
     "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
     "remote": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
     "dest": "comfy/loras", "gb": 1.5},

    # ---- cuda24g: Qwen-Image-Edit-2511 GGUF Q4_K_M (24GB VRAM / 32GB RAM) ----
    # Requires ComfyUI-GGUF. Put .gguf in models/unet, keep fp8 text encoder + VAE.
    {"groups": ["image-gguf"], "kind": "ms_file",
     "repo": "unsloth/Qwen-Image-Edit-2511-GGUF",
     "remote": "qwen-image-edit-2511-Q4_K_M.gguf",
     "dest": "comfy/unet", "gb": 12.8},
    {"groups": ["image-gguf"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "dest": "comfy/text_encoders", "gb": 9.38},
    {"groups": ["image-gguf"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/vae/qwen_image_vae.safetensors",
     "dest": "comfy/vae", "gb": 0.25},

    # ---- cuda24g: Qwen-Image 20B T2I GGUF Q4_K_M (one-time sheets/scenes) ----
    {"groups": ["image-t2i-gguf"], "kind": "ms_file",
     "repo": "unsloth/Qwen-Image-GGUF",
     "remote": "qwen-image-Q4_K_M.gguf",
     "dest": "comfy/unet", "gb": 12.84},
    {"groups": ["image-t2i-gguf"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "dest": "comfy/text_encoders", "gb": 9.38},
    {"groups": ["image-t2i-gguf"], "kind": "ms_file",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "remote": "split_files/vae/qwen_image_vae.safetensors",
     "dest": "comfy/vae", "gb": 0.25},

    # ---- video draft: Wan2.2 TI2V-5B ----
    {"groups": ["video-5b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
     "dest": "comfy/diffusion_models", "gb": 10.0},
    {"groups": ["video-5b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/text_encoders/umt5_xxl_fp16.safetensors",
     "dest": "comfy/text_encoders", "gb": 11.37},
    {"groups": ["video-5b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/vae/wan2.2_vae.safetensors",
     "dest": "comfy/vae", "gb": 1.41},

    # ---- cuda24g video: 5B fp16 + fp8 UMT5 text encoder (lower RAM/VRAM) ----
    {"groups": ["video-5b-24g"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
     "dest": "comfy/diffusion_models", "gb": 10.0},
    {"groups": ["video-5b-24g"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "dest": "comfy/text_encoders", "gb": 6.74},
    {"groups": ["video-5b-24g"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/vae/wan2.2_vae.safetensors",
     "dest": "comfy/vae", "gb": 1.41},

    # ---- video final: Wan2.2 I2V-A14B fp8 + LightX2V 4-step ----
    {"groups": ["video-14b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
     "dest": "comfy/diffusion_models", "gb": 14.29},
    {"groups": ["video-14b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
     "dest": "comfy/diffusion_models", "gb": 14.29},
    {"groups": ["video-14b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
     "dest": "comfy/loras", "gb": 1.23},
    {"groups": ["video-14b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
     "dest": "comfy/loras", "gb": 1.23},
    {"groups": ["video-14b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "dest": "comfy/text_encoders", "gb": 6.74},
    {"groups": ["video-14b"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/vae/wan_2.1_vae.safetensors",
     "dest": "comfy/vae", "gb": 0.25},

    # ---- optional: Wan2.2-S2V (audio-driven talking video) ----
    {"groups": ["s2v"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors",
     "dest": "comfy/diffusion_models", "gb": 16.4},
    {"groups": ["s2v"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors",
     "dest": "comfy/audio_encoders", "gb": 0.63},
    {"groups": ["s2v"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "dest": "comfy/text_encoders", "gb": 6.74},
    {"groups": ["s2v"], "kind": "ms_file",
     "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
     "remote": "split_files/vae/wan_2.1_vae.safetensors",
     "dest": "comfy/vae", "gb": 0.25},

    # ---- TTS ----
    {"groups": ["tts"], "kind": "ms_snapshot",
     "repo": "iic/CosyVoice2-0.5B", "dest": "cosyvoice2", "gb": 2.0},

    # ---- LLM for storyboard ----
    {"groups": ["llm"], "kind": "ms_snapshot",
     "repo": "Qwen/Qwen3-8B", "dest": "qwen3-8b", "gb": 16.0},
    {"groups": ["llm-small"], "kind": "ms_snapshot",
     "repo": "Qwen/Qwen3-4B", "dest": "qwen3-4b", "gb": 8.0},

    # ---- ASR / subtitle alignment (optional) ----
    {"groups": ["asr"], "kind": "ms_snapshot",
     "repo": "iic/SenseVoiceSmall", "dest": "sensevoice", "gb": 1.0},

    # ---- optional upscale model ----
    {"groups": ["upscale"], "kind": "url",
     "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
     "name": "RealESRGAN_x4plus_anime_6B.pth",
     "dest": "comfy/upscale_models", "gb": 0.017},
]


PROFILES = {
    "mi300x": "image,image-t2i,video-5b,tts,llm-small,asr",
    "cuda24g": "image-gguf,image-t2i-gguf,video-5b-24g,tts,llm-small,asr",
    "cuda24g-min": "image-gguf,video-5b-24g,tts,llm-small,asr",
}


def human(n: float) -> str:
    if n >= 1024:
        return f"{n / 1024:.1f}TB"
    return f"{n:.1f}GB"


def selected(groups: list[str], entry: dict) -> bool:
    if "all" in groups:
        return True
    return any(g in groups for g in entry["groups"])


def expected_path(entry: dict) -> Path:
    if entry["kind"] in ("ms_file", "hf_file"):
        return DEST / entry["dest"] / Path(entry["remote"]).name
    if entry["kind"] == "ms_snapshot":
        return DEST / entry["dest"]
    if entry["kind"] == "url":
        return DEST / entry["dest"] / entry["name"]
    raise ValueError(entry["kind"])


def size_gb(path: Path) -> float:
    if path.is_symlink():
        target = path.resolve(strict=False)
        if not target.exists():
            return 0.0
        path = target
    if path.is_file():
        return path.stat().st_size / 1e9
    if path.is_dir():
        total = 0
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        return total / 1e9
    return 0.0


def check_size(entry: dict, path: Path, quiet: bool = False) -> bool:
    actual = size_gb(path)
    expected = float(entry["gb"])
    tolerance = 0.60 if expected < 2.0 else 0.15
    ok = abs(actual - expected) <= expected * tolerance
    if not quiet:
        flag = "OK " if ok else "WARN"
        print(f"  [{flag}] {path.name}: {actual:.2f}GB (expect ~{expected}GB)")
    return ok


def link_or_copy(src: Path, dest: Path) -> None:
    src = src.resolve(strict=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        if dest.resolve(strict=False) == src:
            return
        dest.unlink()
    elif dest.exists():
        dest.unlink()
    dest.symlink_to(src)


def download_ms_file(entry: dict) -> Path:
    from modelscope.hub.file_download import model_file_download
    CACHE.mkdir(parents=True, exist_ok=True)
    local = model_file_download(
        model_id=entry["repo"],
        file_path=entry["remote"],
        local_dir=str(CACHE / "ms" / entry["repo"].replace("/", "__")),
    )
    src = Path(local)
    dest = expected_path(entry)
    link_or_copy(src, dest)
    return src


def download_ms_snapshot(entry: dict) -> Path:
    from modelscope import snapshot_download
    dest = expected_path(entry)
    dest.mkdir(parents=True, exist_ok=True)
    # snapshot_download is incremental: re-running resumes a partial snapshot.
    # Do not early-return just because the directory is non-empty.
    snapshot_download(model_id=entry["repo"], local_dir=str(dest))
    return dest


def download_hf_file(entry: dict) -> Path:
    from huggingface_hub import hf_hub_download
    CACHE.mkdir(parents=True, exist_ok=True)
    src = Path(hf_hub_download(
        repo_id=entry["repo"],
        filename=entry["remote"],
        local_dir=str(CACHE / "hf" / entry["repo"].replace("/", "__")),
    ))
    dest = expected_path(entry)
    link_or_copy(src, dest)
    return src


def download_url(entry: dict) -> Path:
    import requests
    dest = expected_path(entry)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(entry["url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return dest


def do_download(entry: dict) -> Path:
    kind = entry["kind"]
    if kind == "ms_file":
        return download_ms_file(entry)
    if kind == "ms_snapshot":
        return download_ms_snapshot(entry)
    if kind == "hf_file":
        return download_hf_file(entry)
    if kind == "url":
        return download_url(entry)
    raise ValueError(kind)


def main() -> int:
    global DEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="image,video-5b",
                    help="comma separated: image,image-t2i,image-gguf,image-t2i-gguf,video-5b,video-5b-24g,video-14b,s2v,tts,llm,llm-small,asr,upscale,image-bf16,image-lora,all")
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None,
                    help="preset group set: mi300x, cuda24g, cuda24g-min (overrides --groups)")
    ap.add_argument("--dest", default=str(DEST))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the >95GB warning")
    args = ap.parse_args()

    DEST = Path(args.dest)
    if args.profile:
        groups = PROFILES[args.profile].split(",")
        print(f"profile: {args.profile}")
    else:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    entries = [e for e in MANIFEST if selected(groups, e)]
    # de-duplicate (e.g. umt5/vae shared by video-14b and s2v)
    seen = set()
    uniq = []
    for e in entries:
        key = (e["kind"], e["repo"], e.get("remote") or e.get("url") or e["dest"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    entries = uniq

    total = sum(float(e["gb"]) for e in entries)
    print(f"groups: {','.join(groups)}")
    print(f"files : {len(entries)}")
    print(f"size  : ~{human(total)}")
    print()

    if args.list:
        for e in entries:
            name = e.get("remote") or e.get("url") or e["repo"]
            print(f"- [{','.join(e['groups'])}] {e['kind']:12s} {e['repo']:45s} {name}  ~{e['gb']}GB")
        return 0

    if args.verify:
        bad = 0
        for e in entries:
            p = expected_path(e)
            if not p.exists() and not p.is_symlink():
                print(f"  [MISS] {p}")
                bad += 1
                continue
            if not check_size(e, p):
                bad += 1
        print(f"\nverify done for groups={','.join(groups)}, problems: {bad}")
        if len(groups) != 1 or groups[0] != "all":
            print("hint: use --groups all --verify to check every manifest entry")
        return 0 if bad == 0 else 1

    if args.dry_run:
        for e in entries:
            print(f"- {expected_path(e)}  ~{e['gb']}GB")
        return 0

    if total > 95 and not args.yes:
        print("WARNING: selected size is >95GB; 100GB persistent disk will be very tight.")
        print("         consider: --groups image,video-5b (Set A) or put 5B/LLM on /root.")
        print("         re-run with --yes to continue anyway.")
        return 2

    DEST.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(entries, 1):
        dest = expected_path(e)
        print(f"\n[{i}/{len(entries)}] {e['kind']}  {e['repo']}  -> {dest}")
        if (dest.exists() or dest.is_symlink()) and size_gb(dest) > 0.01:
            if check_size(e, dest):
                print("  skip (already complete)")
                continue
            print("  size mismatch, re-running download (resumes where possible)")
        try:
            src = do_download(e)
            check_size(e, src if src.is_file() else dest)
        except Exception as exc:
            print(f"  ERROR: {exc!r}")
            print("  hint: open the repo page and confirm the file path:")
            print(f"        https://www.modelscope.cn/models/{e['repo']}")
            return 1

    print("\ndownload done.")
    print("next: bash scripts/03_start_comfyui.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
