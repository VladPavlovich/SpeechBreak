"""
Live web demo: talk to Kimi-Audio-7B-Instruct, unsteered or steered, in your browser.

  - speak into the microphone or type; Kimi hears / reads it directly and answers out loud
    (spoken turns go through Kimi's own audio encoders; no text-to-speech or speech-to-text step)
  - steering: unsteered, one of the settings in results_kimi/best_settings.json, or any layer / alpha
  - multi-turn: the conversation continues until you press "New conversation"
  - nothing is saved (see scripts/live_common.py)

The steering directions were found on typed prompts; they are applied the same way to spoken ones.

Runs in the `kimi` conda env (see README). Usage (from the repo root):
  conda run -n kimi --no-capture-output python scripts/live_kimi.py          # http://localhost:7861
or: bash scripts/live.sh kimi

The page is served on 127.0.0.1 of the GPU server only. From your laptop, open an SSH tunnel and
browse to http://localhost:7861 (the microphone only works on localhost or https):
  ssh -L 7861:localhost:7861 <you>@<gpu-server>
"""

import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("LOGURU_LEVEL", "WARNING")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import live_common as ui  # noqa: E402  (sets up the private temp folder before gradio loads)
import torch  # noqa: E402

from steering import run_steps_kimi as R  # noqa: E402
from steering.hooks import add_hooks  # noqa: E402
from steering.kimi_audio import KimiAudioModel  # noqa: E402

BEST_SETTINGS = os.path.join(REPO_ROOT, "results_kimi", "best_settings.json")


class KimiBackend:
    title = "Kimi-Audio — live steering"
    intro = ("Speak or type; Kimi answers out loud. Kimi sometimes answers in Chinese; saying or typing "
             "*Answer in English* first may help.")
    streams = {"LLM (writes the text)": "llm", "Audio head (makes the speech)": "audio_head"}
    max_layer, default_layer = 27, 19

    def __init__(self):
        best = json.load(open(BEST_SETTINGS))
        self.presets = {}
        for s in best["jailbreak"]:
            self.presets[f"Jailbreak: LLM layer {s['layer']}, α = {s['alpha']:+g}"] = (s["layer"], s["alpha"])
        for s in best["refuse"]:
            self.presets[f"Refuse: LLM layer {s['layer']}, α = {s['alpha']:+g}"] = (s["layer"], s["alpha"])
        print("Loading Kimi-Audio ...", flush=True)
        self.model, self.dirs = KimiAudioModel(load_detokenizer=True, audio_input=True), R.load_directions()

    preset_spec = staticmethod(R.llm_spec)

    def user_turn(self, mic_path, text):
        """A spoken turn is encoded here (audio tokens + Whisper features, in memory), a typed one tokenized."""
        with torch.inference_mode():
            return self.model.user_turn(wav_path=mic_path) if mic_path else self.model.user_turn(text=text)

    def assistant_turn(self, text):
        return self.model.assistant_turn(text)  # earlier replies are given back as their text

    def reply(self, turns, spec, alpha, max_seconds):
        hooks = R.steering_hooks(self.model, self.dirs, spec, alpha) if spec and alpha else []
        torch.manual_seed(random.randrange(2 ** 31))
        prompt = self.model.conversation_prompt(turns)
        with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
            out = self.model.generate(max_seconds=max_seconds, prompt=prompt)
        return out["audio"].numpy(), out["text"].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to listen on every interface")
    p.add_argument("--share", action="store_true", help="also create a public gradio.live link (anyone with it can use the model)")
    args = p.parse_args()
    ui.launch(KimiBackend(), args.host, args.port, args.share)


if __name__ == "__main__":
    main()
