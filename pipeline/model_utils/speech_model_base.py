"""
SpeechModelBase — abstract base for end-to-end speech models in SpeechBreak.

Key differences from ModelBase:
  - Input is audio (wav tensor or path), not token ids
  - Output is audio tokens + optional text (model-dependent)
  - Block modules may come from multiple residual streams (e.g. Thinker/Talker)
  - generate_completions operates on audio datasets, not instruction strings
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import gc
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks


class SpeechModelBase(ABC):
    """
    Abstract base class for speech models.

    Subclasses must implement all abstract methods.
    The hook infrastructure from hook_utils.py is fully compatible —
    hooks attach to nn.Module objects, which all three model architectures expose.
    """

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path

        # Load model and processor (speech models use processors, not tokenizers)
        self.model = self._load_model(model_name_or_path)
        self.processor = self._load_processor(model_name_or_path)

        # Architecture introspection — populated by subclass
        # For single-stream models (CSM, Moshi): primary_block_modules is the full stack
        # For dual-stream models (Qwen2.5-Omni): primary = Thinker, secondary = Talker
        self.primary_block_modules = self._get_primary_block_modules()
        self.secondary_block_modules = self._get_secondary_block_modules()  # None if single-stream

        self.primary_attn_modules = self._get_primary_attn_modules()
        self.primary_mlp_modules = self._get_primary_mlp_modules()

        # For hook_utils compatibility: expose model_block_modules pointing at primary
        self.model_block_modules = self.primary_block_modules
        self.model_attn_modules = self.primary_attn_modules
        self.model_mlp_modules = self.primary_mlp_modules
        self.model_post_attn_modules = self._get_post_attn_modules()

        # Hidden dims — needed for DIM vector shape validation
        self.primary_hidden_size = self._get_primary_hidden_size()
        self.secondary_hidden_size = self._get_secondary_hidden_size()  # None if single-stream

        # Architecture metadata — used in inspect script and paper
        self.architecture_type = self._get_architecture_type()  # "dual_stream" | "single_stream" | "pure_audio"
        self.has_text_bottleneck = self._has_text_bottleneck()

    # ------------------------------------------------------------------ #
    #  Abstract: loading                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> nn.Module:
        """Load and return the model. Move to appropriate device."""
        pass

    @abstractmethod
    def _load_processor(self, model_name_or_path: str) -> Any:
        """Load processor/tokenizer appropriate for this model."""
        pass

    # ------------------------------------------------------------------ #
    #  Abstract: architecture introspection                                #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _get_primary_block_modules(self) -> List[nn.Module]:
        """
        Return list of transformer block modules for the primary (semantic) stream.
        For Qwen2.5-Omni: Thinker layers.
        For CSM-1B: backbone layers.
        For Moshi: shared LM layers.
        """
        pass

    @abstractmethod
    def _get_secondary_block_modules(self) -> Optional[List[nn.Module]]:
        """
        Return list of transformer block modules for the secondary (acoustic) stream.
        For Qwen2.5-Omni: Talker layers.
        For CSM-1B: audio decoder layers.
        For Moshi: None (single shared stream).
        """
        pass

    @abstractmethod
    def _get_primary_attn_modules(self) -> List[nn.Module]:
        pass

    @abstractmethod
    def _get_primary_mlp_modules(self) -> List[nn.Module]:
        pass

    @abstractmethod
    def _get_post_attn_modules(self) -> Optional[List[nn.Module]]:
        pass

    @abstractmethod
    def _get_primary_hidden_size(self) -> int:
        pass

    @abstractmethod
    def _get_secondary_hidden_size(self) -> Optional[int]:
        pass

    @abstractmethod
    def _get_architecture_type(self) -> str:
        """Returns 'dual_stream', 'single_stream', or 'pure_audio'."""
        pass

    @abstractmethod
    def _has_text_bottleneck(self) -> bool:
        """Whether a text representation exists between audio input and audio output."""
        pass

    # ------------------------------------------------------------------ #
    #  Abstract: audio processing                                          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess audio file into model inputs.
        Returns dict suitable for model forward pass.
        """
        pass

    @abstractmethod
    def generate_audio(
        self,
        inputs: Dict[str, torch.Tensor],
        fwd_pre_hooks: List[Tuple] = [],
        fwd_hooks: List[Tuple] = [],
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Run steered inference. Returns dict with at minimum:
          - 'audio': output wav tensor or path
          - 'text': decoded text if model has text bottleneck, else None
          - 'hidden_states': list of captured activations if hooks fired, else None
        """
        pass

    # ------------------------------------------------------------------ #
    #  Concrete: activation extraction                                     #
    # ------------------------------------------------------------------ #

    def extract_activations(
        self,
        audio_paths: List[str],
        layer_indices: List[int],
        stream: str = "primary",          # "primary" | "secondary"
        token_position: str = "last",     # "last" | "mean" | int
        batch_size: int = 1,
        progress: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract residual stream activations from specified layers.

        Returns dict mapping layer_index -> Tensor[n_samples, hidden_size].

        This is the entry point for DIM vector computation (Step 5).
        Speech models process one audio file at a time (batch_size=1 default)
        because audio lengths vary and padding introduces artifacts.
        """
        block_modules = (
            self.primary_block_modules
            if stream == "primary"
            else self.secondary_block_modules
        )
        if block_modules is None:
            raise ValueError(f"No {stream} stream available for {self.__class__.__name__}")

        # Storage: layer_idx -> list of activation tensors
        activation_store: Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is (hidden_states, ...) for transformer blocks
                hidden = output[0] if isinstance(output, tuple) else output
                # hidden: [batch, seq_len, d_model]
                if token_position == "last":
                    act = hidden[:, -1, :].detach().cpu()
                elif token_position == "mean":
                    act = hidden.mean(dim=1).detach().cpu()
                elif isinstance(token_position, int):
                    act = hidden[:, token_position, :].detach().cpu()
                else:
                    raise ValueError(f"Unknown token_position: {token_position}")
                # Overwrite instead of append — we only want the final forward pass
                # per audio file, not every intermediate generation step
                activation_store[layer_idx] = [act]
            return hook_fn

        fwd_hooks = [
            (block_modules[l], make_hook(l))
            for l in layer_indices
            if l < len(block_modules)
        ]

        progress_iter = tqdm(audio_paths, desc=f"Extracting [{stream}]", disable=not progress)

        for audio_path in progress_iter:
            inputs = self.process_audio(audio_path)
            with torch.inference_mode():
                with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
                    self.generate_audio(inputs)

            torch.cuda.empty_cache()
            gc.collect()

        # Stack per layer: [n_samples, hidden_size]
        return {
            l: torch.cat(acts, dim=0)
            for l, acts in activation_store.items()
            if acts
        }

    # ------------------------------------------------------------------ #
    #  Concrete: batch conversation generation                             #
    # ------------------------------------------------------------------ #

    def run_conversation(
        self,
        audio_path: str,
        fwd_pre_hooks: List[Tuple] = [],
        fwd_hooks: List[Tuple] = [],
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Run a single-turn audio conversation with optional steering hooks.
        Returns the generate_audio output dict.
        """
        inputs = self.process_audio(audio_path)
        with torch.inference_mode():
            with add_hooks(
                module_forward_pre_hooks=fwd_pre_hooks,
                module_forward_hooks=fwd_hooks,
            ):
                result = self.generate_audio(inputs, max_new_tokens=max_new_tokens)
        return result

    def del_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def describe_architecture(self) -> str:
        """Print a summary of hookable components. Used by inspect script."""
        lines = [
            f"Model: {self.model_name_or_path}",
            f"Architecture type: {self.architecture_type}",
            f"Text bottleneck: {self.has_text_bottleneck}",
            f"Primary stream layers: {len(self.primary_block_modules)}",
            f"Primary hidden size: {self.primary_hidden_size}",
        ]
        if self.secondary_block_modules is not None:
            lines += [
                f"Secondary stream layers: {len(self.secondary_block_modules)}",
                f"Secondary hidden size: {self.secondary_hidden_size}",
            ]
        else:
            lines.append("Secondary stream: None (single-stream model)")
        return "\n".join(lines)