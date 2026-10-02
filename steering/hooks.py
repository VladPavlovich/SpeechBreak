"""PyTorch hook helpers: temporarily attach hooks, and add a steering vector to a block's input."""

import contextlib
from typing import Callable, List, Tuple

import torch


@contextlib.contextmanager
def add_hooks(module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
              module_forward_hooks: List[Tuple[torch.nn.Module, Callable]]):
    """Register (module, fn) forward pre-hooks and forward hooks for the duration of the block."""
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            handles.append(module.register_forward_pre_hook(hook))
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def add_vector_hook(vector: torch.Tensor, multiplier: float = 1.0):
    """Forward pre-hook: residual stream (block input) += multiplier * vector, at every position."""
    def hook_fn(module, inputs):
        hidden = inputs[0] if isinstance(inputs, tuple) else inputs
        hidden = hidden + multiplier * vector.to(hidden)
        return (hidden, *inputs[1:]) if isinstance(inputs, tuple) else hidden
    return hook_fn
