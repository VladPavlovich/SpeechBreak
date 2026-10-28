"""
Live web demo: talk to Qwen2.5-Omni-7B, unsteered or steered, in your browser.

  - speak into the microphone or type; Qwen hears / reads it directly and answers out loud
    (no text-to-speech or speech-to-text step in between)
  - steering: unsteered, one of the settings in results/best_settings.json, or any layer / alpha
  - multi-turn: the conversation continues until you press "New conversation"
  - nothing is saved (see scripts/live_common.py)

Runs in the main env (see README). Usage (from the repo root):
  python scripts/live_qwen.py                 # http://localhost:7860
or: bash scripts/live.sh qwen

The page is served on 127.0.0.1 of the GPU server only. From your laptop, open an SSH tunnel and
browse to http://localhost:7860 (the microphone only works on localhost or https):
  ssh -L 7860:localhost:7860 <you>@<gpu-server>
"""

import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import live_common as ui  # noqa: E402  (sets up the private temp folder before gradio loads)
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from steering import run_steps as R  # noqa: E402
from steering.hooks import add_hooks  # noqa: E402
from steering.qwen_omni import SYSTEM_PROMPT, QwenOmni  # noqa: E402

BEST_SETTINGS = os.path.join(REPO_ROOT, "results", "best_settings.json")


class QwenBackend:
    title = "Qwen2.5-Omni — live steering"
    intro = "Speak or type; Qwen answers out loud."
    streams = {"Thinker (writes the text)": "thinker", "Talker (makes the speech)": "talker"}
    max_layer, default_layer = 27, 18

    def __init__(self):
        best = json.load(open(BEST_SETTINGS))
        self.presets = {}
        for s in best["jailbreak"]:
            self.presets[f"Jailbreak: Thinker layer {s['layer']}, α = {s['alpha']:+g}"] = (s["layer"], s["alpha"])
        for s in best["refuse"]:
            self.presets[f"Refuse: Thinker layer {s['layer']}, α = {s['alpha']:+g}"] = (s["layer"], s["alpha"])
        print("Loading Qwen2.5-Omni ...", flush=True)
        self.model, self.dirs = QwenOmni(), R.load_directions()
        self.sr = self.model.processor.feature_extractor.sampling_rate  # 16 kHz

    preset_spec = staticmethod(R.thinker_spec)

    def user_turn(self, mic_path, text):
        """A spoken turn is kept as its waveform (in memory), a typed one as text."""
        if mic_path:
            return {"role": "user", "wave": librosa.load(mic_path, sr=self.sr, mono=True)[0]}
        return {"role": "user", "text": text}

    def assistant_turn(self, text):
        return {"role": "assistant", "text": text}  # earlier replies are given back as their text

    def inputs(self, turns):
        conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        waves = []
        for t in turns:
            if "wave" in t:
                conversation.append({"role": "user", "content": [{"type": "audio", "audio": "microphone"}]})
                waves.append(t["wave"])
            else:
                conversation.append({"role": t["role"], "content": [{"type": "text", "text": t["text"]}]})
        processor = self.model.processor
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, audio=waves or None, return_tensors="pt", padding=True)
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def reply(self, turns, spec, alpha, max_seconds):
        hooks = R.steering_hooks(self.model, self.dirs, spec, alpha) if spec and alpha else []
        torch.manual_seed(random.randrange(2 ** 31))
        inputs = self.inputs(turns)
        with torch.inference_mode(), add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
            # ~4 text tokens and 50 speech tokens per second of speech: don't write more than is spoken
            out = self.model.generate(inputs, max_new_tokens=int(max_seconds * 4),
                                      talker_max_new_tokens=int(max_seconds * 50))
        audio = out["audio"]
        audio = audio.float().cpu().flatten().numpy() if torch.is_tensor(audio) else np.zeros(0, dtype=np.float32)
        return audio, (out["text"][0] if out["text"] else "").strip()


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to listen on every interface")
    p.add_argument("--share", action="store_true", help="also create a public gradio.live link (anyone with it can use the model)")
    args = p.parse_args()
    ui.launch(QwenBackend(), args.host, args.port, args.share)


if __name__ == "__main__":
    main()
