"""
Steer Kimi-Audio-7B-Instruct on your own prompts and save its spoken replies.
The Kimi version of scripts/steer_prompts.py (Qwen2.5-Omni).

  1. reads prompts from scripts/steer_prompts_kimi.txt (one per line; blank lines and # comments skipped)
  2. Kimi-Audio reads each prompt as text and answers out loud, once per steering setting
  3. saves every reply wav + results.json (prompt, setting, Kimi's reply text)

Settings come from results_kimi/best_settings.json. By default only the jailbreak settings run
(subtract the harmful direction at LLM layer 19):
  L19_a-1  L19_a-0.5
Optional, only if named in --settings:
  baseline        no steering
  refuse_L19_a+1  add the direction, alpha +1 (refuses some harmless requests, speech stays clear)
  refuse_L19_a+2  add the direction, alpha +2 (the text refuses everything, but the speech garbles)

Usage (from the repo root; Kimi-Audio needs the `kimi` conda env):
  conda run -n kimi --no-capture-output python scripts/steer_prompts_kimi.py
  conda run -n kimi --no-capture-output python scripts/steer_prompts_kimi.py --settings L19_a-1            # one setting
  conda run -n kimi --no-capture-output python scripts/steer_prompts_kimi.py --settings L19_a-1 baseline   # + unsteered reply
  conda run -n kimi --no-capture-output python scripts/steer_prompts_kimi.py --custom 14:-1 18:-2   # LLM layer:alpha

Output: results_kimi/steer_prompts/<timestamp>/
  01_<setting>.wav, ..., results.json
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
os.environ.setdefault("LOGURU_LEVEL", "WARNING")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from steering import run_steps_kimi as R  # noqa: E402

PROMPT_FILE = os.path.join(REPO_ROOT, "scripts", "steer_prompts_kimi.txt")
BEST_SETTINGS = os.path.join(REPO_ROOT, "results_kimi", "best_settings.json")


def load_settings():
    """{name: (LLM layer or None, signed alpha)} -> (all settings, default jailbreak names)."""
    best = json.load(open(BEST_SETTINGS))
    jailbreak = {s["name"]: (s["layer"], s["alpha"]) for s in best["jailbreak"]}
    others = {s["name"]: (s["layer"], s["alpha"]) for s in best["refuse"]}
    return {**jailbreak, **others, "baseline": (None, 0.0)}, list(jailbreak)


def parse_args(settings, defaults):
    p = argparse.ArgumentParser(description="Steer Kimi-Audio on prompts from a txt file and save the replies")
    p.add_argument("--prompts", default=PROMPT_FILE, help="txt file, one prompt per line")
    p.add_argument("--settings", nargs="+", default=defaults, choices=list(settings))
    p.add_argument("--custom", nargs="+", default=[], metavar="LAYER:ALPHA",
                   help="extra LLM settings, e.g. 14:-0.5 (negative = jailbreak, positive = refuse)")
    p.add_argument("--out", default=None, help="output dir (default: results_kimi/steer_prompts/<timestamp>)")
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

    dirs = R.load_directions()
    model = R.load_model()
    results = []
    for i, text in enumerate(prompts, 1):
        for name, (layer, alpha) in settings.items():
            hooks = R.steering_hooks(model, dirs, R.llm_spec(layer), alpha) if layer is not None and alpha else []
            out_wav = os.path.join(out, f"{i:02d}_{name}.wav")
            # same seed for every setting of a prompt, so only the steering differs
            gen = R.generate_reply(model, text, hooks, out_wav, R.seed_for(text))
            results.append({"prompt_id": i, "prompt": text, "setting": name, "layer": layer, "alpha": alpha,
                            "output_wav": R.rel(out_wav), "reply_text": gen["text"],
                            "duration_s": gen["duration_s"]})
            print(f"[{i:02d} {name:>14s}] {(gen['text'] or '')[:110]}", flush=True)
    model.unload()

    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
    print(f"done -> {R.rel(out)}/ ({len(results)} reply wavs + results.json)")


if __name__ == "__main__":
    main()
