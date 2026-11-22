"""Spoken versions of SpeechBreak's prompt data.

Two sources of audio:

* **TTS** of the existing text splits / eval sets. Clips are synthesized on
  demand and cached on disk, keyed by (backend, voice, text), so a pipeline run
  only synthesizes the prompts it actually samples and reruns are free.
* **VoiceBench** (hlt-lab/voicebench), which ships spoken prompts, e.g.
  `advbench` (520 harmful) and `alpacaeval` (harmless).

Pre-warm the cache (optional; the pipeline also synthesizes lazily):

    python -m dataset.audio_data --tts mms --splits strongreject_harmful_train harmless_val
    python -m dataset.audio_data --tts mms --datasets advbench jailbreakbench
"""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

DATASET_DIR = os.path.dirname(os.path.realpath(__file__))
AUDIO_CACHE_DIR = os.environ.get("SPEECHBREAK_AUDIO_CACHE", os.path.join(DATASET_DIR, "audio", "cache"))
SR = 16_000
MAX_SECONDS = 30.0  # Whisper-style encoders (Qwen2-Audio) see at most 30 s

MODALITIES = ("text", "audio", "audio+text")


# ----------------------------------------------------------------------------
# TTS backends
# ----------------------------------------------------------------------------
class TTSBackend:
    name: str = "base"
    voice: str = "default"

    def synthesize(self, text: str) -> np.ndarray:  # float32 mono @ 16 kHz
        raise NotImplementedError

    @property
    def key(self) -> str:
        return f"{self.name}:{self.voice}"


class MMSTTS(TTSBackend):
    """facebook/mms-tts-eng (VITS) via transformers: local, ~150 MB, no extra
    system dependencies, outputs 16 kHz. Single speaker. Seeded per clip so a
    given text always produces the same waveform."""

    name = "mms"

    def __init__(self, model_id: str = "facebook/mms-tts-eng", device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.voice = model_id.split("/")[-1]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(self.device).eval()
        self.sr = self.model.config.sampling_rate

    def synthesize(self, text: str) -> np.ndarray:
        import torch
        from pipeline.utils.audio_utils import load_audio
        seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            wav = self.model(**inputs).waveform[0].float().cpu().numpy()
        return load_audio(wav, SR, sampling_rate=self.sr)


class MacSayTTS(TTSBackend):
    """macOS `say`. Handy for quick local checks; not available on Linux GPUs."""

    name = "say"

    def __init__(self, voice: str = "Samantha"):
        self.voice = voice

    def synthesize(self, text: str) -> np.ndarray:
        from pipeline.utils.audio_utils import load_audio
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "clip.wav")
            subprocess.run(["say", "-v", self.voice, "-o", out, "--file-format=WAVE",
                            f"--data-format=LEI16@{SR}", text], check=True)
            return load_audio(out)


_TTS_CACHE: Dict[str, TTSBackend] = {}


def get_tts(spec: Union[str, TTSBackend] = "mms") -> TTSBackend:
    """`mms`, `mms:<hf model id>`, `say` or `say:<voice>`."""
    if isinstance(spec, TTSBackend):
        return spec
    if spec not in _TTS_CACHE:
        name, _, arg = spec.partition(":")
        if name == "mms":
            _TTS_CACHE[spec] = MMSTTS(arg) if arg else MMSTTS()
        elif name == "say":
            _TTS_CACHE[spec] = MacSayTTS(arg) if arg else MacSayTTS()
        else:
            raise ValueError(f"Unknown TTS backend {spec!r} (use mms, mms:<model>, say, say:<voice>)")
    return _TTS_CACHE[spec]


# ----------------------------------------------------------------------------
# Attaching audio to items
# ----------------------------------------------------------------------------
def _clip_path(tts: TTSBackend, text: str, cache_dir: str) -> str:
    h = hashlib.sha1(f"{tts.key}\n{text}".encode()).hexdigest()
    return os.path.join(cache_dir, tts.name, h[:2], f"{h}.wav")


def tts_clip(text: str, tts: Union[str, TTSBackend] = "mms", cache_dir: str = AUDIO_CACHE_DIR) -> str:
    """Path to a cached 16 kHz WAV of `text`, synthesizing it if needed."""
    import soundfile as sf
    tts = get_tts(tts)
    path = _clip_path(tts, text, cache_dir)
    if not os.path.exists(path):
        wav = tts.synthesize(text)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.wav"
        sf.write(tmp, wav, SR, subtype="PCM_16")
        os.replace(tmp, path)  # atomic: parallel runs never see half-written clips
    return path


def as_items(data: Iterable[Union[str, dict]]) -> List[dict]:
    """Normalise a split (list of strings or dicts) to a list of new dicts."""
    return [dict(x) if isinstance(x, dict) else {"instruction": str(x), "category": None} for x in data]


def attach_audio(
    data: Iterable[Union[str, dict]],
    tts: Union[str, TTSBackend] = "mms",
    cache_dir: str = AUDIO_CACHE_DIR,
    progress: bool = True,
) -> List[dict]:
    """Return copies of the items with `audio` set to a TTS clip of their
    instruction. Items that already carry audio (e.g. VoiceBench) are kept."""
    from tqdm import tqdm
    items = as_items(data)
    todo = [it for it in items if it.get("audio") is None]
    for it in tqdm(todo, desc=f"TTS ({get_tts(tts).key})", disable=not progress or not todo):
        it["audio"] = tts_clip(it["instruction"], tts, cache_dir)
    return items


def strip_audio(data: Iterable[Union[str, dict]]) -> List[dict]:
    """Text-only items. Items without audio are returned as-is (strings stay
    strings), so text runs see exactly the data they did before SpeechBreak;
    some text tokenizers (llama2, yi) only accept strings."""
    return [{k: v for k, v in x.items() if k != "audio"} if isinstance(x, dict) and "audio" in x else x
            for x in data]


def apply_modality(data, modality: str, tts: Union[str, TTSBackend] = "mms", **kw) -> List[dict]:
    """Make a split match a modality. "audio+text" items look the same as "audio"
    items; whether the transcript is also shown is a model setting
    (Qwen2AudioModel.audio_text_mode)."""
    assert modality in MODALITIES, f"modality must be one of {MODALITIES}"
    return strip_audio(data) if modality == "text" else attach_audio(data, tts, **kw)


def long_prompt_report(items: List[dict], chars_per_second: float = 15.0) -> int:
    """Rough count of prompts whose speech would exceed the encoder window."""
    n = sum(len(it["instruction"]) / chars_per_second > MAX_SECONDS for it in items)
    if n:
        print(f"[audio] warning: ~{n}/{len(items)} prompts are likely > {MAX_SECONDS:.0f}s of speech "
              f"and will be truncated by the audio encoder")
    return n


# ----------------------------------------------------------------------------
# VoiceBench
# ----------------------------------------------------------------------------
VOICEBENCH_DATASETS = {
    # pipeline dataset name: (hf config, hf split, category)
    "voicebench_advbench": ("advbench", "test", None),
    "voicebench_alpacaeval": ("alpacaeval", "test", None),
}


def load_voicebench(name_or_config: str, split: Optional[str] = None, limit: Optional[int] = None) -> List[dict]:
    """Load a VoiceBench subset as pipeline items:
    {"instruction": prompt text, "category": ..., "audio": {"array", "sampling_rate"}}."""
    import io
    import soundfile as sf
    from datasets import Audio
    from datasets import load_dataset as hf_load_dataset
    from pipeline.utils.audio_utils import load_audio
    if name_or_config in VOICEBENCH_DATASETS:
        config, split, category = VOICEBENCH_DATASETS[name_or_config]
    else:
        config, split, category = name_or_config, split or "test", None
    ds = hf_load_dataset("hlt-lab/voicebench", config, split=split)
    # decode ourselves with soundfile (datasets' decoder would need librosa)
    ds = ds.cast_column("audio", Audio(decode=False))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for row in ds:
        a = row["audio"]
        if a.get("bytes"):
            wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
            wav = load_audio(wav.mean(axis=1), SR, sampling_rate=sr)
        else:
            wav = load_audio(a["path"])
        items.append({"instruction": row["prompt"], "category": category,
                      "audio": {"array": wav, "sampling_rate": SR}})
    return items


# ----------------------------------------------------------------------------
# CLI: pre-warm the TTS cache
# ----------------------------------------------------------------------------
def main():
    from dataset.load_dataset import load_dataset, load_dataset_split
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tts", default="mms")
    ap.add_argument("--splits", nargs="*", default=[], help="e.g. strongreject_harmful_train harmless_val")
    ap.add_argument("--datasets", nargs="*", default=[], help="processed eval sets, e.g. advbench")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items of each (smoke tests)")
    args = ap.parse_args()

    for spec in args.splits:
        harmtype, _, split = spec.rpartition("_")
        items = load_dataset_split(harmtype, split)[: args.limit]
        long_prompt_report(as_items(items))
        attach_audio(items, args.tts)
        print(f"[ok] {spec}: {len(items)} clips")
    for name in args.datasets:
        items = load_dataset(name)[: args.limit]
        long_prompt_report(as_items(items))
        attach_audio(items, args.tts)
        print(f"[ok] {name}: {len(items)} clips")
    print("cache:", AUDIO_CACHE_DIR)


if __name__ == "__main__":
    main()
