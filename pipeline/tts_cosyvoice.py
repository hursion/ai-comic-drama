"""CosyVoice2 wrapper for Chinese dialogue (cross-lingual voice cloning).

Requires transformers==4.51.3 in venv-tts: 5.x changes Qwen2 cache/rotary
semantics and makes CosyVoice2-0.5B emit garbage speech (see
docs/TROUBLESHOOT_TTS.md). Install via scripts/05_install_cosyvoice.sh.

Cross-lingual mode keeps only the timbre of the reference wav and drops prompt
text, so short lines (e.g. "谁？") synthesize cleanly without the zero-shot
"too short than prompt text" degradation. CosyVoice2-0.5B has no spk2info.pt, so
SFT speakers are unavailable; a reference wav is required for normal use.

Single line:
  PYTHONPATH=$PWD/CosyVoice:$PWD/third_party/Matcha-TTS \
    venv-tts/bin/python -m pipeline.tts_cosyvoice \
    --text "又是最后一班。" --prompt-wav assets/voices/lin.wav \
    --out episodes/ep01/audio/test.wav

Whole episode (uses each character's voice_ref):
  PYTHONPATH=$PWD/CosyVoice:$PWD/third_party/Matcha-TTS \
    venv-tts/bin/python -m pipeline.tts_cosyvoice --episode episodes/ep01/episode.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline import config


def load_cosyvoice(model_dir: str):
    from cosyvoice.cli.cosyvoice import CosyVoice2
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
    # transformers loads Qwen2 in bf16 but embeddings are fp32; with fp16=False there
    # is no autocast, so unify the LLM to fp32 to avoid a q_proj dtype mismatch.
    cosyvoice.model.llm.float()
    return cosyvoice


def _save(out: Path, audio, sample_rate: int) -> Path:
    import soundfile as sf
    out.parent.mkdir(parents=True, exist_ok=True)
    # torchaudio 2.11 routes load/save through torchcodec; soundfile is reliable.
    sf.write(str(out), audio.squeeze(0).cpu().numpy(), sample_rate)
    return out


def synth(cosyvoice, text: str, prompt_wav: str, out: Path) -> Path:
    import torch
    # The frontend re-loads prompt_wav internally, so pass the path, not a tensor.
    chunks = [item["tts_speech"] for item in
              cosyvoice.inference_cross_lingual(text, prompt_wav, stream=False)]
    if not chunks:
        raise RuntimeError(f"no audio generated for: {text}")
    audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    return _save(out, audio, cosyvoice.sample_rate)


def synth_sft(cosyvoice, text: str, spk_id: str, out: Path) -> Path:
    """Fallback for models that ship spk2info.pt (NOT CosyVoice2-0.5B)."""
    import torch
    chunks = [item["tts_speech"] for item in cosyvoice.inference_sft(text, spk_id, stream=False)]
    if not chunks:
        raise RuntimeError(f"no audio generated for: {text}")
    audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    return _save(out, audio, cosyvoice.sample_rate)


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
    ap.add_argument("--prompt-text", default=None,
                    help="ignored; cross-lingual mode does not use prompt text")
    ap.add_argument("--prompt-wav", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--sft-speaker", default=None,
                    help="fallback built-in speaker id (requires spk2info.pt; not CosyVoice2-0.5B)")
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
                out = episode_dir / "audio" / f"{shot['id']}_{i}.wav"
                if ref.is_file():
                    print(f"[tts] {shot['id']}_{i} (cross-lingual) -> {out}")
                    synth(cosyvoice, text, str(ref), out)
                else:
                    spk = ch.get("sft_speaker") or args.sft_speaker
                    if not spk:
                        print(f"WARN: no usable voice for {ch['id']} (need voice_ref, or --sft-speaker with spk2info.pt)")
                        continue
                    print(f"[tts] {shot['id']}_{i} (sft:{spk}) -> {out}")
                    synth_sft(cosyvoice, text, spk, out)
                count += 1
        print(f"generated {count} lines")
        return 0

    if not (args.text and args.prompt_wav and args.out):
        print("single-line mode needs --text --prompt-wav --out")
        return 2
    out = synth(cosyvoice, args.text, args.prompt_wav, Path(args.out))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
