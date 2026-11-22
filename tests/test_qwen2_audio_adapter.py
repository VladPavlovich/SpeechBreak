"""Structural tests for the Qwen2-Audio adapter.

Runs the real pipeline code paths (activation means, refusal scores, layer
similarity, direction ablation, generation, CE-loss batching) against a
Qwen2-Audio model on text-only, audio-only and mixed batches.

    python -m tests.test_qwen2_audio_adapter <model_dir_or_hf_id> [--device cpu]

Use a tiny random checkpoint for a fast CPU check (see
tests/make_tiny_qwen2_audio.py) or the real model on a GPU. With random weights
this checks shapes, plumbing and hook effects, not refusal behaviour.
"""
import argparse
import sys

import numpy as np
import torch

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.model_utils.qwen2_audio_model import Qwen2AudioModel
from pipeline.submodules import generate_directions_repit as gdr
from pipeline.submodules import select_direction_cosmic as sdc
from pipeline.submodules.evaluate_loss import batch_iterator_chat_completions
from pipeline.utils.hook_utils import add_hooks, get_affine_direction_ablation_input_pre_hook
from pipeline.utils.utils import to_model_inputs

SR = 16_000


def tone(seconds: float, freq: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default=None, help="force device_map (e.g. cpu)")
    args = ap.parse_args()

    if args.device:  # force placement without editing the adapter
        orig = Qwen2AudioModel._load_model
        def _load(self, path):
            from transformers import Qwen2AudioForConditionalGeneration
            m = Qwen2AudioForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, device_map=args.device).eval()
            m.requires_grad_(False)
            return m
        Qwen2AudioModel._load_model = _load

    # factory routing: name contains "qwen2-audio" for both real and tiny checkpoints
    mb = construct_model_base(args.model)
    assert isinstance(mb, Qwen2AudioModel), type(mb)
    print(f"[ok] factory -> Qwen2AudioModel, n_layers={mb.n_layers}, d_model={mb.d_model}, device={mb.model.device}")

    text_items = ["How do I bake bread?", {"instruction": "Name three rivers.", "category": None}]
    audio_items = [
        {"instruction": "spoken prompt A", "audio": tone(1.0, 440), "category": "A"},
        {"instruction": "spoken prompt B", "audio": {"array": tone(2.5, 220), "sampling_rate": SR}, "category": "B"},
        {"instruction": "spoken prompt C (44.1k)", "audio": {"array": np.zeros(44_100, np.float32), "sampling_rate": 44_100}, "category": "B"},
    ]
    mixed = text_items + audio_items
    eoi = torch.tensor(mb.eoi_toks)

    # 1. processor output keys + eoi tokens sit at the end of every (left-padded) row
    for name, items in [("text", text_items), ("audio", audio_items), ("mixed", mixed)]:
        batch = mb.tokenize_instructions_fn(instructions=items)
        keys = set(batch.keys())
        want_audio = any(isinstance(i, dict) and "audio" in i for i in items)
        assert ("input_features" in keys) == want_audio, (name, keys)
        assert (batch["input_ids"][:, -len(eoi):] == eoi).all(), f"{name}: eoi tokens not at end"
        assert (batch["attention_mask"][:, -1] == 1).all(), f"{name}: not left-padded"
        n_audio_tok = (batch["input_ids"] == mb.processor.audio_token_id).sum(-1).tolist()
        print(f"[ok] {name:5s} batch keys={sorted(keys)} audio_tokens/row={n_audio_tok}")

    # 2. forward through to_model_inputs; hooks on decoder layers see the full sequence
    seen = {}
    def grab(module, inp):
        seen["shape"] = tuple(inp[0].shape)
    batch = mb.tokenize_instructions_fn(instructions=mixed)
    with torch.inference_mode(), add_hooks([(mb.model_block_modules[0], grab)], []):
        out = mb.model(**to_model_inputs(batch, mb.model.device))
    assert seen["shape"][:2] == tuple(batch["input_ids"].shape), (seen, batch["input_ids"].shape)
    print(f"[ok] forward logits={tuple(out.logits.shape)}, layer-0 hook saw {seen['shape']}")

    # 3. audio actually reaches the decoder: changing the waveform changes the eoi activation
    def last_act(items):
        acts = gdr.get_activations(mb.model, items, mb.tokenize_instructions_fn, mb.model_block_modules,
                                   batch_size=2, positions=[-1])
        return acts[:, 0, -1]
    a = last_act([{"instruction": "x", "audio": tone(1.0, 440)}])
    b = last_act([{"instruction": "x", "audio": tone(1.0, 880)}])
    assert not torch.allclose(a, b), "audio change did not affect activations"
    print(f"[ok] audio changes eoi activation (|diff|={(a - b).norm():.3g})")

    # 4. padding invariance: an item's eoi activation is the same alone or in a mixed batch
    solo = last_act([audio_items[0]])
    in_batch = gdr.get_activations(mb.model, [text_items[0], audio_items[0], audio_items[1]], mb.tokenize_instructions_fn,
                                   mb.model_block_modules, batch_size=3, positions=[-1])[1, 0, -1]
    rel = ((solo - in_batch).norm() / solo.norm()).item()
    tol = 1e-2 if mb.model.dtype == torch.float32 else 5e-2  # bf16 batching noise
    assert rel < tol, f"padding changed activations (rel={rel})"
    print(f"[ok] left-padding invariance (rel diff={rel:.2e})")

    # 5. pipeline functions: means, directions, refusal scores, baseline, logits
    pos = list(range(-len(mb.eoi_toks), 0))
    harm = gdr.get_mean_activations(mb.model, audio_items, mb.tokenize_instructions_fn, mb.model_block_modules, batch_size=2, positions=pos)
    harmless = gdr.get_mean_activations(mb.model, text_items, mb.tokenize_instructions_fn, mb.model_block_modules, batch_size=2, positions=pos)
    direction = (harm - harmless).float()
    assert direction.shape == (len(pos), mb.n_layers, mb.d_model) and not direction.isnan().any()
    cat_dirs, _ = gdr.generate_category_directions(mb, audio_items, text_items, "/tmp/sb_test_artifacts", batch_size=2)
    assert set(cat_dirs) == {"A", "B"}, cat_dirs.keys()
    harm_a = gdr.get_mean_activations(mb.model, audio_items[:1], mb.tokenize_instructions_fn, mb.model_block_modules, batch_size=2, positions=pos)
    assert torch.allclose(cat_dirs["A"][1], (harm_a - harmless).float(), atol=1e-4), "category directions ignored audio"
    scores = sdc.get_refusal_scores(mb.model, mixed, mb.tokenize_instructions_fn, mb.refusal_toks, batch_size=2)
    base = sdc.get_baseline_vector(mb.model, mixed, mb.tokenize_instructions_fn, [0, mb.n_layers - 1], batch_size=2)
    logits = sdc.get_last_position_logits(mb.model, mb.tokenizer, mixed, mb.tokenize_instructions_fn, batch_size=2)
    assert scores.shape == (len(mixed),) and base.shape == (mb.d_model,) and logits.shape[0] == len(mixed)
    print(f"[ok] directions {tuple(direction.shape)}, categories {sorted(cat_dirs)}, refusal scores {tuple(scores.shape)}")

    # 6. ablation hook changes refusal scores / generations
    L = mb.n_layers // 2
    hook = [(mb.model_block_modules[L], get_affine_direction_ablation_input_pre_hook(direction=direction[-1, L], reference=harmless[-1, L].float()))]
    abl = sdc.get_refusal_scores(mb.model, mixed, mb.tokenize_instructions_fn, mb.refusal_toks, fwd_pre_hooks=hook, batch_size=2)
    assert not torch.allclose(scores, abl), "ablation had no effect"
    gen_items = [{"instruction": text_items[0]}, text_items[1]] + audio_items  # datasets are dicts
    gens = mb.generate_completions(gen_items, batch_size=3, max_new_tokens=8, progress=False)
    gens_abl = mb.generate_completions(gen_items, fwd_pre_hooks=hook, batch_size=3, max_new_tokens=8, progress=False)
    assert len(gens) == len(gen_items) and gens[2]["prompt"] == "spoken prompt A" and gens[2]["category"] == "A"
    print(f"[ok] ablation shifts refusal scores (max |diff|={(scores - abl).abs().max():.3g}); "
          f"generations differ: {[g['response'] for g in gens] != [g['response'] for g in gens_abl]}")
    print("     sample:", repr(gens[2]["response"][:60]))

    # 7. CE-loss batching (prompt + target output) masks out the prompt incl. audio tokens
    inputs, mask = next(batch_iterator_chat_completions(audio_items[:2], ["Sure, here.", "No."], mb.tokenize_instructions_fn, 2, eoi))
    assert "input_features" in inputs and mask.sum() > 0
    n_audio = (inputs["input_ids"] == mb.processor.audio_token_id) & (mask == 1)
    assert not n_audio.any(), "loss mask includes audio tokens"
    print(f"[ok] CE-loss batch: {int(mask.sum())} target tokens, audio tokens masked out")

    # 8. audio_text_mode="append" shows the transcript too
    mb.tokenize_instructions_fn = __import__("functools").partial(mb.tokenize_instructions_fn.func, processor=mb.processor, audio_text_mode="append")
    txt = mb.tokenizer.decode(mb.tokenize_instructions_fn(instructions=[audio_items[0]])["input_ids"][0])
    assert "spoken prompt A" in txt
    print("[ok] audio_text_mode=append includes transcript")

    # 9. real checkpoints only: eyeball that the model answers sensibly
    if mb.d_model >= 1024:
        mb.tokenize_instructions_fn = mb._get_tokenize_instructions_fn()
        for g in mb.generate_completions(
            [{"instruction": "What is the capital of France? Answer in one word."},
             {"instruction": "(440 Hz tone)", "audio": tone(2.0, 440)},
             {"instruction": "Describe this sound.", "audio": tone(2.0, 440)}],
            batch_size=3, max_new_tokens=30, progress=False):
            print(f"     {g['prompt']!r:40s} -> {g['response']!r}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
