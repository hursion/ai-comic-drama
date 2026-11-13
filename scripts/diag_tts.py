"""Diagnose CosyVoice2 over-long output: inspect text tokenization + ASR check."""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "CosyVoice"))
sys.path.insert(0, os.path.join(os.getcwd(), "third_party", "Matcha-TTS"))

import torch
from cosyvoice.cli.cosyvoice import CosyVoice2

cv = CosyVoice2("models/cosyvoice2", load_jit=False, load_trt=False, fp16=False)
cv.model.llm.float()
fe = cv.frontend
print("text_frontend:", repr(fe.text_frontend), flush=True)
print("tokenizer:", type(fe.tokenizer), flush=True)
for text in ["又是最后一班。", "到站了。"]:
    norm = fe.text_normalize(text, split=True)
    print("norm:", repr(norm), flush=True)
    for t in norm:
        tok, tlen = fe._extract_text_token(t)
        print("  text=", repr(t), "token_len=", int(tlen), flush=True)
        print("  token_ids=", tok.tolist()[:80], flush=True)
print("DONE_DIAG", flush=True)

try:
    import whisper
    m = whisper.load_model("base")
    for f in ["episodes/ep01/audio/s01_0.wav", "assets/voices/lin.wav"]:
        r = m.transcribe(f)
        print("ASR", f, "lang=", r.get("language"), "text=", repr(r.get("text")), flush=True)
    print("DONE_ASR", flush=True)
except Exception as e:
    print("ASR skipped:", repr(e), flush=True)