"""
Kokoro TTS: text -> spoken prompt wavs (used by steering/run_steps.py and scripts/steer_prompts.py).

Runs in the `tts` conda env (kokoro is not installed in the main env):
  conda run -n tts python scripts/tts_batch.py jobs.json
jobs.json: [{"text": str, "voice": "am_michael", "out": "/abs/path.wav"}, ...]
American English, 24 kHz (same settings as dataset/generate_audio.py).
"""

import json
import sys

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

SAMPLE_RATE = 24000


def main():
    jobs = json.load(open(sys.argv[1]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    for i, job in enumerate(jobs, 1):
        chunks = [audio for _, _, audio in tts(job["text"], voice=job["voice"], speed=1.0)]
        audio = np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        sf.write(job["out"], audio, SAMPLE_RATE)
        if i % 50 == 0 or i == len(jobs):
            print(f"[tts] {i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
