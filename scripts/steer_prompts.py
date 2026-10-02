"""
Steer Qwen2.5-Omni on your own prompts and save its spoken replies.

  1. reads prompts from scripts/steer_prompts.txt (one per line; blank lines and # comments skipped)
  2. Kokoro TTS (conda env "tts") turns each prompt into the spoken input
  3. Qwen2.5-Omni answers it once per steering setting
  4. saves every reply wav + results.json (prompt, setting, Qwen's text, optional Whisper transcript)

Settings come from results/best_settings.json. By default only the jailbreak settings run
(subtract the harmful direction at Thinker layer 18 / 19):
  L18_a-0.5  L18_a-1  L19_a-0.5  L19_a-1
Optional, only if named in --settings:
  baseline        no steering
  refuse_L18_a+2  add the direction at Thinker layer 18, alpha +2 (refuses even harmless requests)

Usage (from the repo root):
  python scripts/steer_prompts.py
  python scripts/steer_prompts.py --settings L18_a-1             # just one setting
  python scripts/steer_prompts.py --settings L18_a-1 baseline    # add an unsteered reply for comparison
  python scripts/steer_prompts.py --custom 20:-1 18:-2            # extra Thinker settings, layer:alpha
  python scripts/steer_prompts.py --asr                          # also transcribe the replies (Whisper)

Output: results/steer_prompts/<timestamp>/
  01_input.wav, 01_L18_a-0.5.wav, 01_L18_a-1.wav, 01_L19_a-0.5.wav, 01_L19_a-1.wav, ..., results.json
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from steering import run_steps as R  # noqa: E402
from steering.qwen_omni import QwenOmni  # noqa: E402

PROMPT_FILE = os.path.join(REPO_ROOT, "scripts", "steer_prompts.txt")
BEST_SETTINGS = os.path.join(REPO_ROOT, "results", "best_settings.json")


def load_settings():
    """{name: (Thinker layer or None, signed alpha)} -> (all settings, default jailbreak names)."""
    best = json.load(open(BEST_SETTINGS))
    jailbreak = {s["name"]: (s["layer"], s["alpha"]) for s in best["jailbreak"]}
    others = {s["name"]: (s["layer"], s["alpha"]) for s in best["refuse"]}
    return {**jailbreak, **others, "baseline": (None, 0.0)}, list(jailbreak)


def parse_args(settings, defaults):
    p = argparse.ArgumentParser(description="Steer Qwen2.5-Omni on prompts from a txt file and save the replies")
    p.add_argument("--prompts", default=PROMPT_FILE, help="txt file, one prompt per line")
    p.add_argument("--settings", nargs="+", default=defaults, choices=list(settings))
    p.add_argument("--custom", nargs="+", default=[], metavar="LAYER:ALPHA",
                   help="extra Thinker settings, e.g. 19:-0.5 (negative = jailbreak, positive = refuse)")
    p.add_argument("--voice", default="am_michael", help="Kokoro voice for the spoken prompt")
    p.add_argument("--asr", action="store_true", help="also transcribe every reply with Whisper")
    p.add_argument("--out", default=None, help="output dir (default: results/steer_prompts/<timestamp>)")
    return p.parse_args()


def main():
    all_settings, defaults = load_settings()
    args = parse_args(all_settings, defaults)

    if not os.path.exists(args.prompts):
        raise SystemExit(f"{args.prompts} not found")
    prompts = [l.strip() for l in open(args.prompts) if l.strip() and not l.strip().startswith("#")]
    if not prompts:
        raise SystemExit(f"{R.rel(args.prompts)} has no prompts; write one per line")

    settings = {name: all_settings[name] for name in args.settings}
    for c in args.custom:
        layer, alpha = c.split(":")
        settings[f"L{int(layer)}_a{float(alpha):+g}"] = (int(layer), float(alpha))

    out = args.out or R.path("steer_prompts", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"{len(prompts)} prompt(s) x {len(settings)} setting(s) {list(settings)} -> {R.rel(out)}", flush=True)

    # 1. text -> speech
    inputs = [os.path.join(out, f"{i:02d}_input.wav") for i in range(1, len(prompts) + 1)]
    R.speak(prompts, inputs, args.voice)

    # 2. spoken prompt -> steered Qwen speech
    dirs = R.load_directions()
    model = QwenOmni()
    results = []
    for i, (text, in_wav) in enumerate(zip(prompts, inputs), 1):
        for name, (layer, alpha) in settings.items():
            hooks = R.steering_hooks(model, dirs, R.thinker_spec(layer), alpha) if layer is not None and alpha else []
            out_wav = os.path.join(out, f"{i:02d}_{name}.wav")
            # same seed for every setting of a prompt, so only the steering differs
            gen = R.generate_reply(model, in_wav, hooks, out_wav, R.seed_for(text))
            results.append({"prompt_id": i, "prompt": text, "setting": name, "layer": layer, "alpha": alpha,
                            "input_wav": R.rel(in_wav), "output_wav": R.rel(out_wav),
                            "thinker_text": gen["text"], "duration_s": gen["duration_s"]})
            print(f"[{i:02d} {name:>14s}] {(gen['text'] or '')[:110]}", flush=True)
    model.unload()

    # 3. optional: reply wav -> text (Whisper)
    if args.asr:
        from steering import scoring
        asr = scoring.transcribe_and_embed([os.path.join(REPO_ROOT, r["output_wav"]) for r in results])
        for r in results:
            r["output_asr"] = asr[os.path.join(REPO_ROOT, r["output_wav"])]["asr"]

    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
    print(f"done -> {R.rel(out)}/ ({len(results)} reply wavs + results.json)")


if __name__ == "__main__":
    main()
