import torch
import einops
from torch import Tensor
from jaxtyping import Int, Float

def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = einops.einsum(matrix, vec.unsqueeze(-1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj

# ---------------------------------------------------------------------------
# Modality-agnostic model helpers.
#
# Everything that runs a forward pass goes through `to_model_inputs`, so a
# tokenizer (text) or a processor (audio: input_features, feature masks, ...)
# can both feed the pipeline.  Layer count / width are read through
# `get_text_config`, because multimodal models keep them on `config.text_config`.
# ---------------------------------------------------------------------------

_NON_MODEL_KEYS = ("token_type_ids",)


def to_model_inputs(batch, device) -> dict:
    """Turn a tokenizer/processor output into kwargs for `model(**...)` / `model.generate(**...)`."""
    out = {}
    for k, v in dict(batch).items():
        if k in _NON_MODEL_KEYS or v is None:
            continue
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def get_text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def get_n_layers(model) -> int:
    return get_text_config(model).num_hidden_layers


def get_d_model(model) -> int:
    return get_text_config(model).hidden_size


def empty_device_cache(synchronize: bool = False):
    """CUDA/MPS-safe replacement for torch.cuda.empty_cache()."""
    if torch.cuda.is_available():
        if synchronize:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        if synchronize:
            torch.mps.synchronize()
        torch.mps.empty_cache()
