"""
dataset/generate_audio.py

Speak every "instruction" of the two prompt sets with Kokoro-82M TTS. These wavs are the spoken
requests the steering directions are computed from (steering/data.py):

  dataset/processed/harmbench_illegal_harassment_misinfo.json -> dataset/audio/harmful_audio/   (155 HarmBench)
  dataset/processed/harmless_alpaca.json                      -> dataset/audio/harmless_audio/  (985 Alpaca)

Entry i is spoken with voice ALL_VOICES[i % 14] and saved as {i:04d}_{category}.wav (24 kHz mono),
plus manifest.json (entries + audio_path, voice, spoken_text, duration_s) in each folder.
Bracket placeholders such as "[Victim's Name]" are replaced before speaking (see spoken_text()).

Re-running skips existing wavs, so it can resume.

Kokoro needs Python <= 3.12 (spaCy), so it runs from its own conda env:
  conda create -n tts python=3.12 -y && conda run -n tts pip install kokoro soundfile tqdm

Usage (from the repo root):
  conda run -n tts --no-capture-output python dataset/generate_audio.py
  conda run -n tts --no-capture-output python dataset/generate_audio.py --limit 10
"""

import argparse
import json
import os
import re

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from tqdm import tqdm

SAMPLE_RATE = 24000
# prompt set -> audio folder
INPUTS = {
    "dataset/processed/harmbench_illegal_harassment_misinfo.json": "dataset/audio/harmful_audio",
    "dataset/processed/harmless_alpaca.json": "dataset/audio/harmless_audio",
}

# American English Kokoro voices (lang_code "a")
ALL_VOICES = ["am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "am_fenrir", "am_puck", "am_echo",
              "af_heart", "af_bella", "af_nicole", "af_sarah", "af_nova", "af_sky"]


def spoken_text(text):
    """Replace placeholders like [Victim's Name] / [Victim_Name]; drop other [..] tags."""
    text = re.sub(r"(Mr\.?/Ms\.?\s*)?\[[^\]]*name[^\]]*\]", "Mr. Johnson", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize(tts, text, voice, speed):
    chunks = [audio for _, _, audio in tts(text, voice=voice, speed=speed)]
    return torch.cat(chunks).numpy() if chunks else np.zeros(0, dtype=np.float32)


def process_file(tts, path, out_dir, speed, limit):
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for idx, entry in enumerate(tqdm(data, desc=os.path.basename(path))):
        wav_path = os.path.join(out_dir, f"{idx:04d}_{entry.get('category', 'item')}.wav")
        voice = ALL_VOICES[idx % len(ALL_VOICES)]
        text = spoken_text(entry["instruction"])
        if not os.path.exists(wav_path):
            sf.write(wav_path, synthesize(tts, text, voice, speed), SAMPLE_RATE)
        manifest.append({**entry, "audio_path": wav_path, "voice": voice, "spoken_text": text,
                         "duration_s": round(sf.info(wav_path).duration, 3)})

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    durs = [m["duration_s"] for m in manifest]
    print(f"{path}: {len(manifest)} wavs, {sum(durs) / 60:.1f} min total, "
          f"{min(durs):.1f}-{max(durs):.1f}s each -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N entries per file (debug)")
    parser.add_argument("--gpu", type=int, default=None, help="sets CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    for path, out_dir in INPUTS.items():
        process_file(tts, path, out_dir, args.speed, args.limit)


if __name__ == "__main__":
    main()
