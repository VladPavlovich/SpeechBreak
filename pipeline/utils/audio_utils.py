"""Audio loading helpers shared by audio model adapters and data loaders."""
from math import gcd
from typing import Any, Optional

import numpy as np

TARGET_SR = 16_000  # Whisper-style encoders (Qwen2-Audio, Ultravox) expect 16 kHz mono


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    from scipy.signal import resample_poly
    g = gcd(sr, target_sr)
    return resample_poly(wav, target_sr // g, sr // g).astype(np.float32)


def load_audio(src: Any, target_sr: int = TARGET_SR, sampling_rate: Optional[int] = None) -> np.ndarray:
    """Return a mono float32 waveform at `target_sr`.

    Accepts a file path, a HuggingFace-datasets audio dict
    ({"array": ..., "sampling_rate": ...} or {"path": ...}), a numpy array or a
    torch tensor. Bare arrays are assumed to already be at `target_sr` unless
    `sampling_rate` is given.
    """
    if isinstance(src, dict):
        if src.get("array") is not None:
            return load_audio(src["array"], target_sr, src.get("sampling_rate", sampling_rate))
        if src.get("path"):
            return load_audio(src["path"], target_sr)
        raise ValueError(f"Audio dict has neither 'array' nor 'path': {list(src)}")

    if isinstance(src, str):
        import soundfile as sf
        wav, sr = sf.read(src, dtype="float32", always_2d=True)
        return _resample(wav.mean(axis=1), sr, target_sr)

    if hasattr(src, "detach"):  # torch.Tensor
        src = src.detach().cpu().numpy()

    wav = np.asarray(src, dtype=np.float32)
    if wav.ndim == 2:  # (channels, time) or (time, channels): average the short axis
        wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[1] else 1)
    return _resample(wav, sampling_rate or target_sr, target_sr)


def item_audio(item: Any):
    """The audio attached to a dataset item, or None for text-only items."""
    return item.get("audio") if isinstance(item, dict) else None


def item_text(item: Any) -> str:
    """The text part of a dataset item (the instruction / transcript)."""
    if isinstance(item, dict):
        return str(item.get("instruction") or "")
    return str(item)
