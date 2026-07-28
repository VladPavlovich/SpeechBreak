"""
Qwen2.5-Omni-7B wrapper for SpeechBreak.

Architecture: Thinker-Talker dual stream.
  - Thinker: standard transformer processing text + audio tokens semantically
  - Talker: separate transformer that generates audio tokens conditioned on Thinker output
  - Both streams have independent residual streams → hookable separately

HuggingFace model id: Qwen/Qwen2.5-Omni-7B
Requires: pip install transformers>=4.51.0 qwen-omni-utils
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from pipeline.model_utils.speech_model_base import SpeechModelBase


class QwenOmniModel(SpeechModelBase):

    def _load_model(self, model_name_or_path: str) -> nn.Module:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",  # remove if no flash-attn
        )
        model.eval()
        return model

    def _load_processor(self, model_name_or_path: str) -> Any:
        return Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)

    # ------------------------------------------------------------------ #
    #  Architecture introspection                                          #
    # ------------------------------------------------------------------ #
    #
    #  Confirmed attribute paths (Qwen2.5-Omni-7B):
    #    Thinker blocks : model.thinker.model.layers         [28 layers, d=3584]
    #    Thinker attn   : model.thinker.model.layers[i].self_attn
    #    Thinker mlp    : model.thinker.model.layers[i].mlp
    #    Talker blocks  : model.talker.model.layers           [varies]
    #    Talker attn    : model.talker.model.layers[i].self_attn
    #
    #  Run tools/inspect_architecture.py to confirm on your checkpoint.

    def _get_primary_block_modules(self) -> List[nn.Module]:
        """Thinker transformer layers."""
        return list(self.model.thinker.model.layers)  # 28 layers, confirmed

    def _get_secondary_block_modules(self) -> Optional[List[nn.Module]]:
        """Talker transformer layers."""
        return list(self.model.talker.model.layers)  # 24 layers  # 24 layers, confirmed

    def _get_primary_attn_modules(self) -> List[nn.Module]:
        return [layer.self_attn for layer in self.model.thinker.model.layers]  # Qwen2_5OmniAttention

    def _get_primary_mlp_modules(self) -> List[nn.Module]:
        return [layer.mlp for layer in self.model.thinker.model.layers]  # Qwen2MLP

    def _get_post_attn_modules(self) -> Optional[List[nn.Module]]:
        # Post-attention layer norm inside each Thinker block
        return [layer.post_attention_layernorm for layer in self.model.thinker.model.layers]  # Qwen2RMSNorm

    def _get_primary_hidden_size(self) -> int:
        return self.model.thinker.config.text_config.hidden_size  # 3584, confirmed

    def _get_secondary_hidden_size(self) -> Optional[int]:
        return self.model.talker.config.hidden_size  # 896, confirmed

    def _get_architecture_type(self) -> str:
        return "dual_stream"

    def _has_text_bottleneck(self) -> bool:
        # Thinker produces text tokens; Talker consumes them to produce audio
        return True

    # ------------------------------------------------------------------ #
    #  Audio processing                                                    #
    # ------------------------------------------------------------------ #

    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """
        Build model inputs from a wav file path.
        Uses Qwen2.5-Omni's required default system prompt for audio output mode.
        """
        from qwen_omni_utils import process_mm_info

        # Must use exact default system prompt for audio output to work
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)

        # processor expects 'audio' not 'audios' in newer versions
        inputs = self.processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def generate_audio(
        self,
        inputs: Dict[str, torch.Tensor],
        fwd_pre_hooks=None,
        fwd_hooks=None,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate audio response. Returns both audio waveform and Thinker text.
        The Thinker text is used for Thinker-Talker divergence scoring (Phase 6).
        """
        if fwd_pre_hooks is None:
            fwd_pre_hooks = []
        if fwd_hooks is None:
            fwd_hooks = []

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_audio_in_video=False,
            )

        # Qwen2.5-Omni returns (text_ids [1, seq], audio_waveform [samples])
        if isinstance(output, tuple):
            text_ids, audio_output = output[0], output[1]
        else:
            audio_output = output
            text_ids = None

        text_decoded = None
        if text_ids is not None:
            prompt_len = inputs["input_ids"].shape[-1]
            new_ids = text_ids[:, prompt_len:]
            text_decoded = self.processor.tokenizer.batch_decode(
                new_ids, skip_special_tokens=True
            )

        return {
            "audio": audio_output,
            "text": text_decoded,
            "hidden_states": None,  # populated by extract_activations hooks
        }

    # ------------------------------------------------------------------ #
    #  Qwen-specific: get secondary (Talker) block modules for hooking     #
    # ------------------------------------------------------------------ #

    def get_talker_block_modules(self) -> List[nn.Module]:
        """Convenience accessor for Talker layers used in cross-transfer experiments."""
        return list(self.model.talker.model.layers)  # 24 layers  # 24 layers, confirmed

    def get_talker_attn_modules(self) -> List[nn.Module]:
        return [layer.self_attn for layer in self.model.talker.model.layers]  # confirmed