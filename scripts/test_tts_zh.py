import os, sys, time
import torch
import soundfile as sf
import transformers
print("transformers", transformers.__version__, "torch", torch.__version__, flush=True)
sys.path.insert(0, "CosyVoice")
sys.path.insert(0, "third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice2

cv = CosyVoice2("models/cosyvoice2", load_jit=False, load_trt=False, fp16=False)
print("spks", cv.list_available_spks(), "sr", cv.sample_rate, flush=True)
text = "又是最后一班。"
out = "episodes/ep01/audio/test_zh.wav"
os.makedirs(os.path.dirname(out), exist_ok=True)
t0 = time.time()
for r in cv.inference_cross_lingual(text, "assets/voices/lin.wav", stream=False):
    wav = r["tts_speech"].squeeze(0).cpu().numpy()
    sf.write(out, wav, cv.sample_rate)
    dur = len(wav) / cv.sample_rate
    print("saved", out, "dur", round(dur, 2), "elapsed", round(time.time() - t0, 1), flush=True)

import whisper
m = whisper.load_model("base")
res = m.transcribe(out)
print("ASR lang", res.get("language"), "text", res.get("text"), flush=True)