"""Regenerate episode dialogue with a built-in Chinese SFT speaker.

Zero-shot voice cloning produced abnormally long / garbled output for ep01,
so fall back to CosyVoice2's native Chinese speaker for correct pronunciation.

Run on the instance (venv-tts + CosyVoice on PYTHONPATH):
  PYTHONPATH=$PWD/CosyVoice:$PWD/third_party/Matcha-TTS \
    venv-tts/bin/python -u scripts/regen_tts_sft.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.getcwd(), "CosyVoice"))
sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

cv = CosyVoice2("models/cosyvoice2", load_jit=False, load_trt=False, fp16=False)
cv.model.llm.float()
spks = cv.list_available_spks()
print("speakers:", spks, flush=True)
spk = "中文女" if "中文女" in spks else next((x for x in spks if "中文" in x), spks[0])
print("using speaker:", spk, flush=True)

ep = json.load(open("episodes/ep01/episode.json", encoding="utf-8"))
for s in ep["shots"]:
    if not s.get("enabled", True):
        continue
    sid = s["id"]
    for i, d in enumerate(s.get("dialogue") or []):
        text = str(d.get("text", "")).strip()
        if not text:
            continue
        out = "episodes/ep01/audio/" + sid + "_" + str(i) + ".wav"
        chunks = [it["tts_speech"] for it in cv.inference_sft(text, spk, stream=False)]
        audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        torchaudio.save(out, audio, cv.sample_rate)
        dur = audio.shape[1] / cv.sample_rate
        print("[sft] " + sid + "_" + str(i) + " " + repr(text) + " -> " + format(dur, ".2f") + "s", flush=True)
print("DONE", flush=True)