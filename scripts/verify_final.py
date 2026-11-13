import whisper
m = whisper.load_model("base")
r = m.transcribe("/tmp/final_audio.wav")
print("LANG", r["language"], flush=True)
print("TEXT", r["text"], flush=True)
for seg in r["segments"]:
    print(round(seg["start"], 1), round(seg["end"], 1), seg["text"], flush=True)