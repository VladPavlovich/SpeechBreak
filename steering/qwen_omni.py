"""
Qwen2.5-Omni-7B wrapper: audio in, text + speech out, with access to both residual streams.

  Thinker  model.thinker.model.layers   28 layers, d = 3584   (understands the input, writes the text)
  Talker   model.talker.model.layers    24 layers, d = 896    (turns the Thinker's text into speech)
  Bridge   model.talker.thinker_to_talker_proj   3584 -> 896

The Talker reads the Thinker's final hidden states and token embeddings through the bridge,
not just the generated text.
"""

import gc
from typing import Any, Dict, List

import librosa
import torch
import torch.nn as nn
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from steering.hooks import add_hooks

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
STREAMS = ("thinker", "talker")

# Qwen2.5-Omni only produces speech with exactly this system prompt
SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                 "perceiving auditory and visual inputs, as well as generating text and speech.")


class _PrefillCaptured(Exception):
    """Raised from a capture hook to stop generation once the prefill activations are recorded."""


class QwenOmni:

    def __init__(self, model_id: str = MODEL_ID):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # .to() instead of device_map: the full model (~21 GB peak) fits on one GPU
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.device = self.model.device

    # ------------------------------------------------------------------ #
    #  Architecture                                                        #
    # ------------------------------------------------------------------ #

    def blocks(self, stream: str) -> List[nn.Module]:
        """Transformer blocks of the 'thinker' or 'talker' residual stream."""
        return list(self.model.thinker.model.layers if stream == "thinker" else self.model.talker.model.layers)

    @property
    def thinker_to_talker(self) -> torch.Tensor:
        """Weight of the Thinker -> Talker projection, [896, 3584]."""
        return self.model.talker.thinker_to_talker_proj.weight

    # ------------------------------------------------------------------ #
    #  Inputs                                                              #
    # ------------------------------------------------------------------ #

    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """Model inputs for one spoken user turn (a wav file)."""
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
        ]
        wave = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate, mono=True)[0]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=[wave], return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def last_audio_position(self, inputs: Dict[str, torch.Tensor]) -> int:
        """
        Index of the last <|AUDIO|> token. The Talker prefill is position-aligned with the
        Thinker prompt, so the same index addresses the last audio token in both streams.
        """
        audio_token = self.model.config.thinker_config.audio_token_index
        return int((inputs["input_ids"][0] == audio_token).nonzero()[-1].item())

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate(self, inputs: Dict[str, torch.Tensor], max_new_tokens: int = 128,
                 talker_max_new_tokens: int = 1500) -> Dict[str, Any]:
        """
        Spoken reply: {"audio": waveform tensor at 24 kHz, "text": [Thinker text]}.
        max_new_tokens bounds the Thinker's text; talker_max_new_tokens bounds the Talker's
        codec tokens (~50 per second of audio).
        """
        with torch.inference_mode():
            try:
                text_ids, audio = self.model.generate(
                    **inputs, thinker_max_new_tokens=max_new_tokens, talker_max_new_tokens=talker_max_new_tokens,
                    use_audio_in_video=False, return_audio=True)
            except RuntimeError as e:
                # The Talker emitted no codec tokens (e.g. under strong steering) and token2wav
                # crashes on the empty sequence: a silent reply, not an error.
                if "cannot reshape tensor of 0 elements" not in str(e):
                    raise
                text_ids = self.model.generate(**inputs, thinker_max_new_tokens=max_new_tokens,
                                               use_audio_in_video=False, return_audio=False)
                audio = torch.zeros(0)
        new_ids = text_ids[:, inputs["input_ids"].shape[-1]:]
        return {"audio": audio, "text": self.processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)}

    def _run_prefill(self, inputs: Dict[str, torch.Tensor], stream: str):
        """The cheapest generate() call that runs the stream's prefill (aborted from a hook)."""
        if stream == "thinker":
            self.model.generate(**inputs, thinker_max_new_tokens=1, use_audio_in_video=False, return_audio=False)
        else:  # the Talker needs >= 2 Thinker tokens
            self.model.generate(**inputs, thinker_max_new_tokens=2, talker_max_new_tokens=2,
                                use_audio_in_video=False, return_audio=True)

    # ------------------------------------------------------------------ #
    #  Activations                                                         #
    # ------------------------------------------------------------------ #

    def extract_activations(self, audio_paths: List[str], stream: str) -> torch.Tensor:
        """
        Residual stream (block input) at the last audio token of the prompt, from the stream's
        prefill pass, for every layer: [n_files, n_layers, d] (float32, cpu). Generation is
        aborted once the last layer has fired, so no text or audio is produced.
        """
        blocks = self.blocks(stream)
        last = len(blocks) - 1
        state = {"pos": None, "done": set(), "acts": {}}

        def make_hook(layer):
            def hook_fn(module, args):
                if layer in state["done"]:  # only the prefill, not the decode steps
                    return
                state["acts"][layer] = args[0][0, state["pos"], :].detach().float().cpu()
                state["done"].add(layer)
                if layer == last:
                    raise _PrefillCaptured()
            return hook_fn

        hooks = [(b, make_hook(l)) for l, b in enumerate(blocks)]
        out = []
        for path in audio_paths:
            inputs = self.process_audio(path)
            state.update(pos=self.last_audio_position(inputs), done=set(), acts={})
            with torch.inference_mode(), add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
                try:
                    self._run_prefill(inputs, stream)
                except _PrefillCaptured:
                    pass
            if len(state["done"]) != len(blocks):
                raise RuntimeError(f"{path}: {stream} layers {sorted(set(range(len(blocks))) - state['done'])} never ran")
            out.append(torch.stack([state["acts"][l] for l in range(len(blocks))]))
            torch.cuda.empty_cache()
        return torch.stack(out)

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
