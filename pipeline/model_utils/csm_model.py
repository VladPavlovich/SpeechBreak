"""
CSM-1B (Conversational Speech Model) wrapper for SpeechBreak.

Architecture: Two torchtune Llama3.2 transformers.
  - backbone: 16-layer Llama3.2-1B, embed_dim=2048 — processes text+audio context
  - decoder: 4-layer Llama3.2-100M, embed_dim=1024 — generates audio codebook tokens

Real structure (from submodules/csm/models.py + generator.py):
  generator                    : Generator
  generator._model             : Model (nn.Module)
  generator._model.backbone    : torchtune TransformerDecoder (16 layers)
  generator._model.decoder     : torchtune TransformerDecoder (4 layers)
  generator._text_tokenizer    : Llama3 AutoTokenizer
  generator._audio_tokenizer   : Mimi codec

torchtune TransformerDecoder layer path:
  model.backbone.layers[i]     : TransformerDecoderLayer
  model.backbone.layers[i].attn: grouped-query attention
  model.backbone.layers[i].mlp : feedforward block

License: Apache 2.0
"""

from typing import Any, Dict, List, Optional
import sys
import torch
import torch.nn as nn
import torchaudio

from pipeline.model_utils.speech_model_base import SpeechModelBase


class CSMModel(SpeechModelBase):

    SAMPLE_RATE = 24000

    def _load_model(self, model_name_or_path: str) -> Any:
        """
        Load CSM Generator via the repo's own load_csm_1b() function.
        generator._model is the actual nn.Module we hook into.
        """
        csm_path = "submodules/csm"
        if csm_path not in sys.path:
            sys.path.insert(0, csm_path)

        from generator import load_csm_1b

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = load_csm_1b(device=device)
        # Don't call .eval() on Generator — it's not nn.Module
        # Call it on the underlying model instead
        generator._model.eval()
        return generator

    def _load_processor(self, model_name_or_path: str) -> Any:
        """
        Processor is the text tokenizer — already loaded inside Generator.
        We just expose it here for interface compatibility.
        """
        return self.model._text_tokenizer

    # ------------------------------------------------------------------ #
    #  Architecture introspection                                          #
    # ------------------------------------------------------------------ #

    def _get_primary_block_modules(self) -> List[nn.Module]:
        """
        Backbone transformer layers — main semantic residual stream.
        torchtune TransformerDecoder stores layers in .layers
        """
        return list(self.model._model.backbone.layers)

    def _get_secondary_block_modules(self) -> Optional[List[nn.Module]]:
        """
        Audio decoder (Depth/100M transformer) layers.
        """
        return list(self.model._model.decoder.layers)

    def _get_primary_attn_modules(self) -> List[nn.Module]:
        # torchtune uses .attn (not .self_attn) on TransformerDecoderLayer
        layers = self.model._model.backbone.layers
        # Try both common names
        if hasattr(layers[0], 'attn'):
            return [layer.attn for layer in layers]
        elif hasattr(layers[0], 'self_attn'):
            return [layer.self_attn for layer in layers]
        else:
            # Fallback: return first child that looks like attention
            return [list(layer.children())[0] for layer in layers]

    def _get_primary_mlp_modules(self) -> List[nn.Module]:
        layers = self.model._model.backbone.layers
        if hasattr(layers[0], 'mlp'):
            return [layer.mlp for layer in layers]
        elif hasattr(layers[0], 'feed_forward'):
            return [layer.feed_forward for layer in layers]
        else:
            return [list(layer.children())[1] for layer in layers]

    def _get_post_attn_modules(self) -> Optional[List[nn.Module]]:
        layers = self.model._model.backbone.layers
        for attr in ['sa_norm', 'mlp_norm']:
            if hasattr(layers[0], attr):
                return [getattr(layer, attr) for layer in layers]
        return None

    def _get_primary_hidden_size(self) -> int:
        # backbone embed_dim = 2048 (llama3_2_1B config)
        return 2048  # confirmed from inspect: text_embeddings shape (128256, 2048)

    def _get_secondary_hidden_size(self) -> Optional[int]:
        # decoder embed_dim = 1024 (llama3_2_100M config)
        return 1024

    def _get_architecture_type(self) -> str:
        return "single_stream"

    def _has_text_bottleneck(self) -> bool:
        # Backbone processes text tokens; decoder generates audio from backbone states
        return True

    # ------------------------------------------------------------------ #
    #  Audio processing                                                    #
    # ------------------------------------------------------------------ #

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Load wav and prepare as a CSM Segment for use as context."""
        csm_path = "submodules/csm"
        if csm_path not in sys.path:
            sys.path.insert(0, csm_path)
        from generator import Segment

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, self.SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        return {
            "audio_tensor": waveform,           # 1D tensor, 24kHz
            "audio_path": audio_path,
            "segment": Segment(speaker=0, text="", audio=waveform),
        }

    def generate_audio(
        self,
        inputs: Dict[str, Any],
        fwd_pre_hooks=None,
        fwd_hooks=None,
        max_new_tokens: int = 512,
        speaker_id: int = 0,
        text_prompt: str = "Hello.",
    ) -> Dict[str, Any]:
        """
        Generate speech using CSM Generator.generate().
        Hooks are already registered by extract_activations / run_conversation
        before this is called — Generator.generate() calls _model.generate_frame()
        internally which fires the forward pass through backbone/decoder.
        """
        csm_path = "submodules/csm"
        if csm_path not in sys.path:
            sys.path.insert(0, csm_path)

        context = [inputs["segment"]] if inputs.get("segment") else []

        # max_audio_length_ms: rough conversion from tokens
        max_ms = max_new_tokens * 80  # each frame = 80ms at 12.5Hz

        audio_output = self.model.generate(
            text=text_prompt,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=max_ms,
            temperature=0.9,
            topk=50,
        )

        return {
            "audio": audio_output,   # 1D tensor at 24kHz
            "text": None,            # CSM has no text output stream
            "hidden_states": None,
        }