"""
Spoken requests used to find the steering directions.

  harmful   dataset/audio/harmful_audio   HarmBench: illegal, misinformation_disinformation,
                                          harassment_bullying (155 clips)
  harmless  dataset/audio/harmless_audio  Alpaca instructions (985 clips)

Both were spoken with Kokoro TTS. Their manifests list older folder names, which are mapped to the
current folders here (the dataset itself is left untouched).

The train / val / test split is frozen in results/splits.json (lists of wav file names), so every
step and every rerun sees the same items. Without that file a fresh 60/20/20 split per category is
made and saved.
"""

import json
import os
import random
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
HARMFUL_MANIFEST = os.path.join(REPO_ROOT, "dataset", "audio", "harmful_audio", "manifest.json")
HARMLESS_MANIFEST = os.path.join(REPO_ROOT, "dataset", "audio", "harmless_audio", "manifest.json")
SPLITS_FILE = os.path.join(REPO_ROOT, "results", "splits.json")

HARMLESS = "harmless"
HARMFUL_CATEGORIES = ["illegal", "misinformation_disinformation", "harassment_bullying"]

RENAMED_DIRS = [("dataset/audio/scam_harmless/", "dataset/audio/harmless_audio/"),
                ("dataset/audio/harmbench_illegal_harassment_misinfo/", "dataset/audio/harmful_audio/")]

Splits = Dict[str, Dict[str, List[dict]]]  # {split: {category: [item, ...]}}


def item_id(item: dict) -> str:
    return os.path.splitext(os.path.basename(item["audio_path"]))[0]


def _fix_path(item: dict) -> dict:
    for old, new in RENAMED_DIRS:
        if item["audio_path"].startswith(old):
            item["audio_path"] = new + item["audio_path"][len(old):]
    return item


def _load_items() -> Dict[str, List[dict]]:
    """{category: [item, ...]} for the harmful categories and harmless."""
    items = {c: [] for c in HARMFUL_CATEGORIES + [HARMLESS]}
    for it in json.load(open(HARMFUL_MANIFEST)):
        if it["category"] in items:
            items[it["category"]].append(_fix_path(it))
    items[HARMLESS] = [_fix_path(dict(it, category=HARMLESS)) for it in json.load(open(HARMLESS_MANIFEST))]
    return items


def load_splits(seed: int = 8) -> Splits:
    items = _load_items()
    if not os.path.exists(SPLITS_FILE):
        rng = random.Random(seed)
        frozen = {"train": {}, "val": {}, "test": {}}
        for cat, its in items.items():
            names = sorted(os.path.basename(it["audio_path"]) for it in its)
            rng.shuffle(names)
            n = len(names)
            frozen["train"][cat] = names[: int(0.6 * n)]
            frozen["val"][cat] = names[int(0.6 * n): int(0.8 * n)]
            frozen["test"][cat] = names[int(0.8 * n):]
        os.makedirs(os.path.dirname(SPLITS_FILE), exist_ok=True)
        json.dump(frozen, open(SPLITS_FILE, "w"), indent=1)
    frozen = json.load(open(SPLITS_FILE))
    by_name = {os.path.basename(it["audio_path"]): it for its in items.values() for it in its}
    return {split: {cat: [by_name[n] for n in names] for cat, names in cats.items()}
            for split, cats in frozen.items()}


def held_out_items(splits: Splits, n: int, offset: int = 0) -> Dict[str, List[dict]]:
    """Val items [offset:offset+n]: {"target": harmful items (categories mixed), "harmless": harmless items}."""
    val = splits["val"]
    harmful = [it for c in HARMFUL_CATEGORIES for it in val.get(c, [])]
    random.Random(0).shuffle(harmful)
    return {"target": harmful[offset:offset + n], "harmless": val[HARMLESS][offset:offset + n]}
