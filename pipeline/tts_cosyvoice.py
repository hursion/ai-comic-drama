"""CosyVoice2 wrapper for zero-shot Chinese dialogue.

Install (in a separate venv is safer, but this works in the ComfyUI venv too):
  git clone --recursive https://github.com/QwenAudio/CosyVoice.git
  cd CosyVoice && pip install -r requirements.txt
  # model:
  python scripts/02_download_models.py --groups tts

Single line:
  python -m pipeline.tts_cosyvoice \
    --text "又是最后一班。" \
    --prompt-text "参考音色的台词文本。" \
    --prompt-wav assets/voices/lin.wav \
    --out episodes/ep01/audio/test.wav

Whole episode:
  python -m pipeline.tts_cosyvoice --episode episodes/ep01/episode.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline import config


def load_cosyvoice(model_dir: str):
    from cosyvoice.cli.cosyvoice import CosyVoice2
    return CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)


def synth(cosyvoice, text: str, prompt_text: str, prompt_wav: str, out: Path) -> Path:
    import torch
    import torchaudio
    from cosyvoice.utils.file_utils import load_wav
    prompt_16k = load_wav(prompt_wav, 16000)
    chunks = []
    for item in cosyvoice.inference_zero_shot(text, prompt_text, prompt_16k, stream=False):
        chunks.append(item["tts_speech"])
    if not chunks:
        raise RuntimeError(f"no audio generated for: {text}")
    audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), audio, cosyvoice.sample_rate)
    return out


def synth_sft(cosyvoice, text: str, spk_id: str, out: Path) -> Path:
    """Fallback when no reference wav is available: use a built-in speaker."""
    import torch
    import torchaudio
    chunks = [item["tts_speech"] for item in cosyvoice.inference_sft(text, spk_id, stream=False)]
    if not chunks:
        raise RuntimeError(f"no audio generated for: {text}")
    audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), audio, cosyvoice.sample_rate)
    return out


def resolve(path_str: str, episode_dir: Path | None = None) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    if episode_dir and (episode_dir / p).exists():
        return episode_dir / p
    return config.WORK / p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cosyvoice2")
    ap.add_argument("--text", default=None)
    ap.add_argument("--prompt-text", default=None)
    ap.add_argument("--prompt-wav", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--sft-speaker", default=None,
                    help="fallback built-in speaker id when a character has no valid voice_ref")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("WARN: torch.cuda is not available; CosyVoice may still run on CPU but will be slow")

    model_dir = args.model
    if not Path(model_dir).is_absolute():
        model_dir = str(config.WORK / model_dir)
    if not Path(model_dir).exists():
        print(f"model dir not found: {model_dir}")
        print("run: python scripts/02_download_models.py --groups tts")
        return 2

    cosyvoice = load_cosyvoice(model_dir)
    print("CosyVoice2 loaded, sample_rate =", cosyvoice.sample_rate)

    if args.episode:
        ep_path = Path(args.episode)
        episode = json.loads(ep_path.read_text(encoding="utf-8"))
        episode_dir = ep_path.parent
        chars = {c["id"]: c for c in episode.get("characters", [])}
        shots = [s for s in episode.get("shots", []) if s.get("enabled", True)]
        count = 0
        for shot in shots[: args.limit or None]:
            for i, line in enumerate(shot.get("dialogue") or []):
                text = str(line.get("text", "")).strip()
                if not text:
                    continue
                ch = chars.get(line.get("character"))
                if not ch:
                    print(f"WARN: {shot['id']} line {i}: unknown character {line.get('character')}")
                    continue
                ref = resolve(ch.get("voice_ref", ""), episode_dir)
                prompt_text = ch.get("voice_text") or args.prompt_text
                out = episode_dir / "audio" / f"{shot['id']}_{i}.wav"
                if ref.is_file() and prompt_text:
                    print(f"[tts] {shot['id']}_{i} (zero-shot) -> {out}")
                    synth(cosyvoice, text, prompt_text, str(ref), out)
                else:
                    spk = ch.get("sft_speaker") or args.sft_speaker
                    if not spk:
                        print(f"WARN: no usable voice for {ch['id']} (need voice_ref+voice_text, or --sft-speaker)")
                        continue
                    print(f"[tts] {shot['id']}_{i} (sft:{spk}) -> {out}")
                    synth_sft(cosyvoice, text, spk, out)
                count += 1
        print(f"generated {count} lines")
        return 0

    if not (args.text and args.prompt_text and args.prompt_wav and args.out):
        print("single-line mode needs --text --prompt-text --prompt-wav --out")
        return 2
    out = synth(cosyvoice, args.text, args.prompt_text, args.prompt_wav, Path(args.out))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
