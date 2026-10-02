"""
dataset/generate_scam_audio.py

Synthesize a .wav for every "instruction" in processed dataset JSONs using
Kokoro-82M TTS (default: scam_harmful.json and scam_harmless.json).

Each scam script (same category + script_id) gets one fixed voice, so the
caller sounds the same across the call. Script-less entries (script_id < 0)
get a voice by index. Bracket placeholders in the text such as
"[Victim's Name]" are replaced before speaking (see spoken_text()).

Output (24 kHz mono), per input file:
  dataset/audio/<input basename>/{idx:04d}_{category}_s{script_id:03d}_t{turn:02d}.wav
  dataset/audio/<input basename>/manifest.json   (entries + audio_path, voice, spoken_text, duration_s)

Re-running skips existing wavs, so it can resume.

Kokoro needs Python <= 3.12 (spaCy), so it runs from its own conda env:
  conda create -n tts python=3.12 -y && conda run -n tts pip install kokoro soundfile tqdm

Usage:
  conda run -n tts --no-capture-output python dataset/generate_scam_audio.py
  conda run -n tts --no-capture-output python dataset/generate_scam_audio.py --limit 10
"""

import os
import re
import json
import argparse

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from kokoro import KPipeline

SAMPLE_RATE = 24000
INPUTS = ["dataset/processed/scam_harmful.json", "dataset/processed/scam_harmless.json"]

# American English Kokoro voices (lang_code "a"). Every scam caller introduces
# himself as "Michael", so scripts use male voices only.
MALE_VOICES = ["am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "am_fenrir", "am_puck", "am_echo"]
ALL_VOICES = MALE_VOICES + ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_nova", "af_sky"]
CATEGORY_OFFSET = {"bank_fraud": 0, "irs_impersonation": 3, "tech_support": 6}


def wav_name(idx, entry):
    sid = entry.get("script_id", -1)
    if sid is None or sid < 0:
        return f"{idx:04d}_{entry.get('category', 'item')}.wav"
    return f"{idx:04d}_{entry['category']}_s{sid:03d}_t{entry['turn']:02d}.wav"


def pick_voice(idx, entry):
    """One voice per scam script; script-less entries cycle by index."""
    sid = entry.get("script_id", -1)
    if sid is None or sid < 0:
        return ALL_VOICES[idx % len(ALL_VOICES)]
    return MALE_VOICES[(sid + CATEGORY_OFFSET.get(entry.get("category"), 0)) % len(MALE_VOICES)]


def spoken_text(text):
    """Replace generator placeholders like [Victim's Name] / [Victim_Name]; drop other [..] tags."""
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
    out_dir = out_dir or os.path.join("dataset", "audio", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for idx, entry in enumerate(tqdm(data, desc=os.path.basename(path))):
        wav_path = os.path.join(out_dir, wav_name(idx, entry))
        voice = pick_voice(idx, entry)
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
    parser.add_argument("--inputs", nargs="+", default=INPUTS)
    parser.add_argument("--out_dir", default=None, help="only valid with a single input")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="only the first N entries per file (debug)")
    parser.add_argument("--gpu", type=int, default=None, help="sets CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.out_dir and len(args.inputs) > 1:
        parser.error("--out_dir needs a single --inputs file")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    for path in args.inputs:
        process_file(tts, path, args.out_dir, args.speed, args.limit)


if __name__ == "__main__":
    main()
