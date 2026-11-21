"""Phase 3 checks: TTS cache, VoiceBench loader, modality-aware dataset
loading, and audio items flowing through direction extraction / selection /
generation.

    python -m tests.test_voice_data <qwen2-audio model dir or id> [--device cpu] [--tts mms]

Uses small sample fractions so it runs on CPU in a few minutes. Downloads
facebook/mms-tts-eng (~150 MB) and VoiceBench advbench on first run.
"""
import argparse
import os
import tempfile

import numpy as np
import soundfile as sf
import torch

import dataset.audio_data as ad
from dataset.load_dataset import load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tts", default="mms")
    args = ap.parse_args()
    ad.AUDIO_CACHE_DIR = tempfile.mkdtemp(prefix="sb_audio_")  # don't touch the real cache

    # 1. TTS: 16 kHz mono, cached, deterministic
    p1 = ad.tts_clip("How do I bake sourdough bread at home?", args.tts, ad.AUDIO_CACHE_DIR)
    wav, sr = sf.read(p1)
    assert sr == 16_000 and wav.ndim == 1 and 0.5 < len(wav) / sr < 10, (sr, wav.shape)
    mtime = os.path.getmtime(p1)
    assert ad.tts_clip("How do I bake sourdough bread at home?", args.tts, ad.AUDIO_CACHE_DIR) == p1
    assert os.path.getmtime(p1) == mtime, "cache miss on second call"
    print(f"[ok] TTS {ad.get_tts(args.tts).key}: {len(wav) / sr:.2f}s @ {sr} Hz, cached")

    # 2. apply_modality
    items = [{"instruction": "Name a river.", "category": "x"}, "Plain string prompt"]
    a = ad.apply_modality(items, "audio", args.tts, cache_dir=ad.AUDIO_CACHE_DIR)
    assert all(os.path.exists(i["audio"]) for i in a) and a[1]["category"] is None
    t = ad.apply_modality(a, "text")
    assert all("audio" not in i for i in t) and "audio" in a[0], "should copy, not mutate"
    raw = ["a string", {"instruction": "a dict", "category": "c"}]
    assert all(x is y for x, y in zip(ad.apply_modality(raw, "text"), raw)), "text mode must not touch text data"
    print("[ok] apply_modality: text <-> audio, inputs not mutated")

    # 3. VoiceBench
    vb = load_dataset("voicebench_advbench")
    assert len(vb) == 520 and vb[0]["audio"]["sampling_rate"] and len(vb[0]["instruction"]) > 10
    vb_text = load_dataset("voicebench_advbench", instructions_only=True)
    assert vb_text[0] == vb[0]["instruction"]
    dur = len(vb[0]["audio"]["array"]) / vb[0]["audio"]["sampling_rate"]
    print(f"[ok] voicebench_advbench: {len(vb)} items, first clip {dur:.1f}s: {vb[0]['instruction'][:50]!r}")

    # 4. Config + modality-aware sampling through the real loader
    from pipeline.run_pipeline_repit import BUILTIN_TARGET_GROUPS, get_category_groups_by_name, load_and_sample_repit_datasets
    cfg = Config(model_alias="test", model_path=args.model, modality="audio", train_modality="text", tts=args.tts,
                 p_train=0.01, p_val=0.03, p_test=0.003)
    assert "voicebench_advbench" in cfg.evaluation_datasets
    try:
        Config(model_alias="x", model_path="x", modality="audio", train_modality="audio+text")
        raise AssertionError("mixing audio and audio+text should fail")
    except ValueError:
        pass
    targets, excl = get_category_groups_by_name("hate")
    assert targets == BUILTIN_TARGET_GROUPS["hate"]
    (h_train, hl_train, h_val, nt_val, hl_val, h_test, hl_test) = load_and_sample_repit_datasets(
        cfg, "hate", targets, exclude=excl, seed=8)
    assert all("audio" not in x for x in h_train + hl_train + h_val + hl_val), "train/val should be text"
    assert h_test and all(os.path.exists(x["audio"]) for x in h_test + hl_test), "test should be audio"
    assert all(isinstance(x, str) for x in hl_train), "text-mode harmless items must stay strings"

    # text-only run: the loader returns exactly what it did before SpeechBreak
    cfg_t = Config(model_alias="t", model_path="x", p_train=0.01, p_val=0.03, p_test=0.003)
    from dataset.load_dataset import load_dataset_split
    orig_apply = ad.apply_modality
    ad.apply_modality = lambda data, *a, **k: list(data)  # bypass = pre-SpeechBreak behaviour
    try:
        before = load_and_sample_repit_datasets(cfg_t, "hate", targets, exclude=excl, seed=8)
    finally:
        ad.apply_modality = orig_apply
    after = load_and_sample_repit_datasets(cfg_t, "hate", targets, exclude=excl, seed=8)
    assert before == after, "text-mode data changed"
    print("[ok] text modality returns data identical to the pre-SpeechBreak loader")
    print(f"[ok] loader (train=text, test=audio): train {len(h_train)}+{len(hl_train)}, "
          f"val {len(h_val)}, test {len(h_test)}+{len(hl_test)} spoken")

    # 5. audio items through the pipeline on the model
    if args.device:
        from tests.test_qwen2_audio_adapter import Qwen2AudioModel
        from transformers import Qwen2AudioForConditionalGeneration
        def _load(self, path):
            m = Qwen2AudioForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, device_map=args.device).eval()
            m.requires_grad_(False)
            return m
        Qwen2AudioModel._load_model = _load
    mb = construct_model_base(args.model, modality="audio")
    assert mb.audio_text_mode == "audio_only"

    cfg_a = Config(model_alias="test-audio", model_path=args.model, modality="audio", tts=args.tts,
                   p_train=0.01, p_val=0.03, p_test=0.003, save_dir=tempfile.mkdtemp(prefix="sb_runs_"))
    (h_train, hl_train, h_val, nt_val, hl_val, h_test, hl_test) = load_and_sample_repit_datasets(
        cfg_a, "hate", targets, exclude=excl, seed=8)
    assert all(os.path.exists(x["audio"]) for x in h_train + hl_train + h_val + hl_val + h_test)

    from pipeline.run_pipeline_repit import compute_and_save_similarity, generate_and_save_category_directions
    from pipeline.submodules.completions_helper import generate_and_save_completions_for_dataset
    from pipeline.submodules.disentangle_vectors import isolate_target_vectors
    cat_dirs, harmless_mean = generate_and_save_category_directions(cfg_a, mb, h_train, hl_train, batch_size=4)
    assert harmless_mean.shape == (len(mb.eoi_toks), mb.n_layers, mb.d_model)
    layers = compute_and_save_similarity(mb.model, cfg_a, h_train, hl_train, mb.tokenize_instructions_fn,
                                         batch_size=4, fraction=0.5)
    cand = isolate_target_vectors(cat_dirs, targets=[t for t in targets if t in cat_dirs], rho=0.0)
    print(f"[ok] audio train set -> {len(cat_dirs)} category directions, layers {layers}, candidates {tuple(cand.shape)}")

    vb_small = ad.apply_modality(load_dataset("voicebench_advbench")[:3], "audio")
    generate_and_save_completions_for_dataset(cfg_a, mb, [], [], "smoke", "voicebench_advbench",
                                              batch_size=3, dataset=vb_small)
    generate_and_save_completions_for_dataset(cfg_a, mb, [], [], "smoke", "harmful", batch_size=4, dataset=h_test[:4])
    out = os.path.join(cfg_a.artifact_path(), "completions")
    print(f"[ok] generations written: {sorted(os.listdir(out))}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
