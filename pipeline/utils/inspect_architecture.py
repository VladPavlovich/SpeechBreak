"""
pipeline/utils/inspect_architecture.py

Run this to confirm attribute paths before any DIM extraction.

Usage:
    PYTHONPATH=. python pipeline/utils/inspect_architecture.py --model csm
    PYTHONPATH=. python pipeline/utils/inspect_architecture.py --model qwen_omni
    PYTHONPATH=. python pipeline/utils/inspect_architecture.py --model moshi
"""

import argparse
import torch
import torch.nn as nn


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_module_tree(module: nn.Module, prefix: str = "", max_depth: int = 3, depth: int = 0):
    if depth > max_depth:
        return
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        own_params = {n: tuple(p.shape) for n, p in child.named_parameters(recurse=False)}
        shape_str = f"  {own_params}" if own_params else ""
        print(f"  {'  ' * depth}{full_name} [{child.__class__.__name__}]{shape_str}")
        print_module_tree(child, full_name, max_depth, depth + 1)


def inspect_csm():
    print_header("CSM-1B Architecture Inspection")

    import sys
    sys.path.insert(0, "submodules/csm")
    from generator import load_csm_1b

    print("\nLoading CSM-1B (downloads weights if not cached)...")
    gen = load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")
    model = gen._model

    print("\n--- Generator attributes ---")
    print(f"  gen._model         : {type(model)}")
    print(f"  gen._text_tokenizer: {type(gen._text_tokenizer)}")
    print(f"  gen._audio_tokenizer: {type(gen._audio_tokenizer)}")
    print(f"  gen.sample_rate    : {gen.sample_rate}")

    print("\n--- Model top-level children ---")
    for name, child in model.named_children():
        param_count = sum(p.numel() for p in child.parameters())
        print(f"  model.{name} [{child.__class__.__name__}] ({param_count/1e6:.1f}M params)")

    print("\n--- Backbone ---")
    print(f"  model.backbone.layers: {len(model.backbone.layers)} layers")
    print(f"  Layer type: {type(model.backbone.layers[0]).__name__}")
    print("\n  Layer 0 children:")
    print_module_tree(model.backbone.layers[0], max_depth=1)

    print("\n--- Decoder ---")
    print(f"  model.decoder.layers: {len(model.decoder.layers)} layers")
    print(f"  Layer type: {type(model.decoder.layers[0]).__name__}")
    print("\n  Layer 0 children:")
    print_module_tree(model.decoder.layers[0], max_depth=1)

    print("\n--- Embedding shapes ---")
    print(f"  text_embeddings : {model.text_embeddings.weight.shape}")
    print(f"  audio_embeddings: {model.audio_embeddings.weight.shape}")
    print(f"  projection      : {model.projection.weight.shape}  (backbone_dim -> decoder_dim)")

    print("\n--- Hook attachment test ---")
    captured = {}
    def test_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["shape"] = tuple(h.shape)

    handle = model.backbone.layers[0].register_forward_hook(test_hook)

    dummy_tokens = torch.zeros(1, 4, 33).long().to(gen.device)
    dummy_mask   = torch.zeros(1, 4, 33).bool().to(gen.device)
    dummy_pos    = torch.arange(4).unsqueeze(0).to(gen.device)
    gen._model.setup_caches(1)

    with torch.inference_mode():
        try:
            gen._model.generate_frame(dummy_tokens, dummy_mask, dummy_pos, temperature=1.0, topk=1)
            if "shape" in captured:
                print(f"  Hook fired on backbone.layers[0]")
                print(f"  Captured shape: {captured['shape']}  (expect [1, seq_len, 2048])")
            else:
                print("  WARNING: hook did not fire — layer path may be wrong")
        except Exception as e:
            print(f"  Forward pass error: {e}")

    handle.remove()
    print("\nCSM-1B inspection complete.")


def inspect_qwen_omni():
    print_header("Qwen2.5-Omni — run after CSM is confirmed")
    print("  PYTHONPATH=. python pipeline/utils/inspect_architecture.py --model qwen_omni")

'''
def inspect_moshi():
    print_header("Moshi — run after Qwen is confirmed")
    print("  PYTHONPATH=. python pipeline/utils/inspect_architecture.py --model moshi")
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["csm", "qwen_omni", "moshi", "all"], default="csm")
    args = parser.parse_args()

    if args.model in ("csm", "all"):
        try:
            inspect_csm()
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {e}")
            traceback.print_exc()
'''
    if args.model in ("qwen_omni", "all"):
        inspect_qwen_omni()

    if args.model in ("moshi", "all"):
        inspect_moshi()
'''

if __name__ == "__main__":
    main()