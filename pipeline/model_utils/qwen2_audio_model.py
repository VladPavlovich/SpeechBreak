"""Qwen2-Audio adapter (e.g. Qwen/Qwen2-Audio-7B-Instruct).

Qwen2-Audio = Whisper-style audio encoder -> projector -> Qwen2 decoder. The
audio embeddings replace <|AUDIO|> placeholder tokens inside the user turn, so
the last tokens of every prompt are still the text "end of instruction"
tokens (`<|im_end|>\\n<|im_start|>assistant\\n`). RepIt's direction
extraction / ablation therefore hooks the *decoder* layers exactly as for a
text Qwen model.

Dataset items may be:
  * a string                         -> text-only prompt
  * {"instruction": str}             -> text-only prompt
  * {"instruction": str, "audio": a} -> spoken prompt. The audio goes in the
    user turn. `instruction` is kept for judging/logging but is NOT shown to
    the model unless `audio_text_mode="append"` (audio + text both).
  `a` is anything `pipeline.utils.audio_utils.load_audio` accepts.
"""
import functools
from typing import List, Optional

import torch
from transformers import AutoProcessor, GenerationConfig, Qwen2AudioForConditionalGeneration

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.audio_utils import TARGET_SR, item_audio, item_text, load_audio

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
AUDIO_PLACEHOLDER = "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
QWEN2_AUDIO_EOI = "<|im_end|>\n<|im_start|>assistant\n"
QWEN2_AUDIO_REFUSAL_TOKS = [40, 2121]  # ['I', 'As'], same Qwen2 tokenizer as text Qwen


def format_qwen2_audio_prompt(text: str, has_audio: bool, system: Optional[str], output: Optional[str] = None) -> str:
    prompt = ""
    if system is not None:
        prompt += f"<|im_start|>system\n{system}<|im_end|>\n"
    prompt += "<|im_start|>user\n"
    if has_audio:
        prompt += AUDIO_PLACEHOLDER
    prompt += text + QWEN2_AUDIO_EOI
    if output is not None:
        prompt += output
    return prompt


def tokenize_instructions_qwen2_audio(
    processor,
    instructions,
    outputs: Optional[List[str]] = None,
    system: Optional[str] = None,
    audio_text_mode: str = "audio_only",
    **_,
):
    """Build processor inputs for a batch. Returns a BatchFeature with
    input_ids, attention_mask and, if any item has audio, input_features and
    feature_attention_mask."""
    if isinstance(instructions, (str, dict)):
        instructions = [instructions]
    # The pipeline passes system=None explicitly in many places; Qwen2-Audio is
    # trained with a system turn, so None means "default". Pass "" to omit it.
    system = DEFAULT_SYSTEM_PROMPT if system is None else (system or None)

    prompts, audios = [], []
    for idx, item in enumerate(instructions):
        audio = item_audio(item)
        has_audio = audio is not None
        if has_audio:
            audios.append(load_audio(audio))
            text = item_text(item) if audio_text_mode == "append" else ""
        else:
            text = item_text(item)
        prompts.append(format_qwen2_audio_prompt(
            text, has_audio, system, output=outputs[idx] if outputs is not None else None))

    kwargs = dict(text=prompts, return_tensors="pt", padding=True)
    if audios:
        kwargs.update(audio=audios, sampling_rate=TARGET_SR)
    return processor(**kwargs)


class Qwen2AudioModel(ModelBase):
    """`audio_text_mode`: "audio_only" (default; spoken prompt only) or
    "append" (spoken prompt + its transcript as text)."""

    def __init__(self, model_name_or_path: str, audio_text_mode: str = "audio_only", dtype=torch.bfloat16):
        assert audio_text_mode in ("audio_only", "append"), audio_text_mode
        self.audio_text_mode = audio_text_mode
        self._dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        super().__init__(model_name_or_path)

    def _load_model(self, model_path):
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self._dtype, device_map="auto",
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
        return tokenizer

    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(do_sample=False)

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen2_audio,
            processor=self.processor,
            audio_text_mode=self.audio_text_mode,
        )

    def _batch_items(self, batch):
        return list(batch)  # keep whole items so the processor sees item["audio"]

    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN2_AUDIO_EOI, add_special_tokens=False)

    def _get_refusal_toks(self):
        return QWEN2_AUDIO_REFUSAL_TOKS

    def _decoder(self):
        return self.model.language_model.model

    def _get_model_block_modules(self):
        return self._decoder().layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([b.self_attn for b in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([b.mlp for b in self.model_block_modules])

    def _get_post_attn_modules(self):
        return torch.nn.ModuleList([b.post_attention_layernorm for b in self.model_block_modules])
