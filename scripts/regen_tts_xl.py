"""Regenerate ep01 dialogue with CosyVoice2 cross-lingual mode.

Earlier garbled output root cause: venv-tts had transformers 5.x, incompatible
with CosyVoice2-0.5B (official pin transformers==4.51.3). The Qwen2 text encoder
cache/rotary semantics changed in 5.x -> garbage speech tokens. 4.51.3 fixes it.

CosyVoice2-0.5B has no spk2info.pt (SFT speakers empty). Cross-lingual keeps only
timbre from the reference wav; prompt_wav must be a PATH string in this code
version. Short Chinese lines synthesize cleanly.

Run on the instance:
  PYTHONPATH=$PWD/CosyVoice:$PWD/third_party/Matcha-TTS \
    venv-tts/bin/python -u scripts/regen_tts_xl.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.getcwd(), "CosyVoice"))
sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "Matcha-TTS"))

import torch
import soundfile as sf
from cosyvoice.cli.cosyvoice import CosyVoice2

cv = CosyVoice2("models/cosyvoice2", load_jit=False, load_trt=False, fp16=False)
print("sample_rate:", cv.sample_rate, flush=True)

ep = json.load(open("episodes/ep01/episode.json", encoding="utf-8"))
ref = "assets/voices/lin.wav"
for s in ep["shots"]:
    if not s.get("enabled", True):
        continue
    sid = s["id"]
    for i, d in enumerate(s.get("dialogue") or []):
        text = str(d.get("text", "")).strip()
        if not text:
            continue
        out = "episodes/ep01/audio/" + sid + "_" + str(i) + ".wav"
        chunks = [it["tts_speech"] for it in cv.inference_cross_lingual(text, ref, stream=False)]
        audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        wav = audio.squeeze(0).cpu().numpy()
        sf.write(out, wav, cv.sample_rate)
        dur = len(wav) / cv.sample_rate
        print("[xl] " + sid + "_" + str(i) + " " + repr(text) + " -> " + format(dur, ".2f") + "s", flush=True)
print("DONE", flush=True)