"""
Moshi wrapper for SpeechBreak.

Architecture: Pure audio token LM — no text bottleneck.
  - Single shared residual stream processes both understanding and generation
  - Parallel inner/outer model: outer LM generates semantic tokens,
    inner "Depth" transformer generates fine-grained codec tokens per step
  - This is the hardest model to probe because everything lives in audio token space

HuggingFace model id: kyutai/moshi
Requires: pip install moshi  (kyutai's own package, not HF transformers)
Docs: https://github.com/kyutai-labs/moshi

WARNING: Moshi's inference stack is non-standard. Attribute paths below
are best-guess from reading the source — MUST be confirmed with inspect script
before running DIM extraction.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from pipeline.model_utils.speech_model_base import SpeechModelBase


class MoshiModel(SpeechModelBase):

    SAMPLE_RATE = 24000
    FRAME_RATE = 12.5  # Moshi processes audio at 12.5 Hz semantic token rate

    def _load_model(self, model_name_or_path: str) -> nn.Module:
        """
        Load Moshi via kyutai's moshi package.
        Returns the LM model object with hookable transformer layers.
        """
        try:
            from moshi.models import loaders
        except ImportError:
            raise ImportError(
                "Moshi package not found. Install with: pip install moshi\n"
                "See: https://github.com/kyutai-labs/moshi"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load via kyutai loader — downloads weights from HF if not cached
        moshi_weight = loaders.hf_hub_download(
            loaders.DEFAULT_REPO, loaders.MOSHI_NAME
        )
        moshi_model = loaders.get_moshi_lm(moshi_weight, device=device)
        moshi_model.eval()
        return moshi_model

    def _load_processor(self, model_name_or_path: str) -> Any:
        """
        Moshi uses a MimiCodec for audio tokenization, not a text tokenizer.
        The codec is loaded separately and used internally during inference.
        """
        try:
            from moshi.models import loaders
        except ImportError:
            raise ImportError("Install moshi: pip install moshi")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mimi_weight = loaders.hf_hub_download(
            loaders.DEFAULT_REPO, loaders.MIMI_NAME
        )
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.eval()
        return mimi  # processor is the codec

    # ------------------------------------------------------------------ #
    #  Architecture introspection                                          #
    # ------------------------------------------------------------------ #
    #
    #  Moshi LM attributes (approximate — verify with inspect script):
    #    Outer LM layers  : model.transformer.layers    [32 layers, d=4096]
    #    Inner LM (Depth) : model.depth_transformer.layers
    #
    #  The outer transformer is what we hook for DIM vectors.
    #  The inner transformer generates codec tokens per outer step.
    #
    #  IMPORTANT: Moshi attribute paths are NOT stable across versions.
    #  Always run inspect_architecture.py after any moshi package update.

    def _get_primary_block_modules(self) -> List[nn.Module]:
        """Outer transformer layers — main audio-semantic residual stream."""
        # Try common attribute paths; adjust if inspect shows different structure
        for attr in ["transformer", "lm_model", "model"]:
            if hasattr(self.model, attr):
                inner = getattr(self.model, attr)
                if hasattr(inner, "layers"):
                    return list(inner.layers)
        raise AttributeError(
            "Could not find transformer layers on Moshi model. "
            "Run inspect_architecture.py and update this method."
        )

    def _get_secondary_block_modules(self) -> Optional[List[nn.Module]]:
        """Depth transformer layers for fine-grained codec token generation."""
        for attr in ["depth_transformer", "decoder"]:
            if hasattr(self.model, attr):
                inner = getattr(self.model, attr)
                if hasattr(inner, "layers"):
                    return list(inner.layers)
        return None  # Return None if depth transformer not separately accessible

    def _get_primary_attn_modules(self) -> List[nn.Module]:
        layers = self._get_primary_block_modules()
        attn_attrs = ["self_attn", "attn", "attention"]
        for attr in attn_attrs:
            if hasattr(layers[0], attr):
                return [getattr(l, attr) for l in layers]
        raise AttributeError("Could not find attention modules on Moshi layers.")

    def _get_primary_mlp_modules(self) -> List[nn.Module]:
        layers = self._get_primary_block_modules()
        mlp_attrs = ["mlp", "feed_forward", "ffn"]
        for attr in mlp_attrs:
            if hasattr(layers[0], attr):
                return [getattr(l, attr) for l in layers]
        raise AttributeError("Could not find MLP modules on Moshi layers.")

    def _get_post_attn_modules(self) -> Optional[List[nn.Module]]:
        layers = self._get_primary_block_modules()
        norm_attrs = ["post_attention_layernorm", "norm2", "post_attn_norm"]
        for attr in norm_attrs:
            if hasattr(layers[0], attr):
                return [getattr(l, attr) for l in layers]
        return None

    def _get_primary_hidden_size(self) -> int:
        # Moshi 7B: hidden_size = 4096
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        # Fallback: infer from first layer weight
        layers = self._get_primary_block_modules()
        for name, param in layers[0].named_parameters():
            if len(param.shape) == 2:
                return param.shape[-1]
        return 4096  # Moshi default

    def _get_secondary_hidden_size(self) -> Optional[int]:
        secondary = self._get_secondary_block_modules()
        if secondary is None:
            return None
        for name, param in secondary[0].named_parameters():
            if len(param.shape) == 2:
                return param.shape[-1]
        return None

    def _get_architecture_type(self) -> str:
        return "pure_audio"  # No text bottleneck anywhere

    def _has_text_bottleneck(self) -> bool:
        return False

    # ------------------------------------------------------------------ #
    #  Audio processing                                                    #
    # ------------------------------------------------------------------ #

    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """
        Encode audio to Mimi codec tokens for model input.
        Moshi operates entirely in codec token space.
        """
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, self.SAMPLE_RATE)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        device = next(self.model.parameters()).device
        waveform = waveform.to(device)

        # Encode to codec tokens using Mimi
        with torch.inference_mode():
            # processor here is the Mimi codec
            codes = self.processor.encode(waveform.unsqueeze(0))  # [1, n_codebooks, T]

        return {
            "audio_codes": codes,
            "audio_waveform": waveform,
            "audio_path": audio_path,
        }

    def generate_audio(
        self,
        inputs: Dict[str, torch.Tensor],
        fwd_pre_hooks=None,
        fwd_hooks=None,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate audio tokens and decode back to waveform.
        Moshi's generation API is streaming — we run it in one shot here.

        NOTE: Moshi's actual inference uses InferenceSession for streaming.
        This is a simplified batch version for benchmarking. Check moshi docs
        for the correct one-shot generation call for your version.
        """
        audio_codes = inputs["audio_codes"]
        device = next(self.model.parameters()).device

        with torch.inference_mode():
            # This call signature may need adjustment based on moshi version
            # Run inspect_architecture.py to confirm generate() signature
            try:
                output_codes = self.model.generate(
                    audio_codes,
                    max_new_tokens=max_new_tokens,
                )
            except TypeError:
                # Fallback: some versions use different signature
                output_codes = self.model(audio_codes)

            # Decode output codes back to waveform via Mimi
            audio_output = self.processor.decode(output_codes)

        return {
            "audio": audio_output,
            "text": None,  # pure audio model — no text output
            "hidden_states": None,
        }