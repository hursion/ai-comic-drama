"""Test CosyVoice2 with fp16=True (autocast) instead of the llm.float() hack."""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "CosyVoice"))
sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

cv = CosyVoice2("models/cosyvoice2", load_jit=False, load_trt=False, fp16=True)
print("fp16:", cv.fp16, flush=True)

text = "又是最后一班。"
ref = "assets/voices/lin.wav"
chunks = [it["tts_speech"] for it in cv.inference_cross_lingual(text, ref, stream=False)]
audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
out = "episodes/ep01/audio/test_fp16.wav"
torchaudio.save(out, audio, cv.sample_rate)
print("duration:", format(audio.shape[1] / cv.sample_rate, ".2f"), flush=True)

import whisper
m = whisper.load_model("base")
r = m.transcribe(out)
print("ASR lang=", r.get("language"), "text=", repr(r.get("text")), flush=True)
print("DONE", flush=True)