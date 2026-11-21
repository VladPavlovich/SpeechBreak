# SpeechBreak: audio-model integration

SpeechBreak extends the RepIt pipeline (concept-specific refusal directions,
see the top-level README) from text-only LLMs to **audio LLMs** such as
Qwen2-Audio, and evaluates them on spoken prompts (VoiceBench-style data).

Audio LLMs are an audio encoder whose output is spliced into the user turn of
an ordinary decoder LLM. RepIt reads and edits activations at the last few
"end-of-instruction" tokens of that decoder, which come *after* the audio, so
direction extraction, selection, ablation and judging carry over unchanged.
Only three things were text-specific: how inputs are built, where layer
count / width live in the config, and the data format.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Modality-agnostic forward passes (`model(**inputs)`) | Done |
| 2 | Audio model adapters (Qwen2-Audio first) | In progress |
| 3 | Voice data: TTS of existing splits, VoiceBench loader, `--modality text\|audio\|both` | Planned |
| 4 | Judging: unchanged (judges read transcript + response) | — |
| 5 | Gradio UI: run pipeline, browse results, try-it with mic upload | Planned |

## Phase 1: modality-agnostic inputs

Every forward pass and `generate` call now goes through one helper:

```python
from pipeline.utils.utils import to_model_inputs
batch = model_base.tokenize_instructions_fn(instructions=items)   # tokenizer OR processor output
out = model(**to_model_inputs(batch, model.device))
```

`to_model_inputs` moves every tensor to the device and drops keys the model
does not accept (`token_type_ids`). Text tokenizers yield
`input_ids, attention_mask`; audio processors additionally yield
`input_features, feature_attention_mask`, which now pass straight through.

Other helpers in `pipeline/utils/utils.py`:

- `get_n_layers(model)`, `get_d_model(model)`: read `config.text_config` when
  present (multimodal models), else `config`.
- `empty_device_cache(synchronize=False)`: CUDA / MPS-safe replacement for
  `torch.cuda.empty_cache()`.

Extension points on `ModelBase` (`pipeline/model_utils/model_base.py`):

- `n_layers`, `d_model` properties.
- `_batch_items(batch)`: what `generate_completions` hands to the tokenize fn.
  Text models get instruction strings; audio models override it to keep whole
  items so the processor can see `item["audio"]`.
- `prepare_inputs(items, system=None)`: tokenize/process and return
  device-ready kwargs.

**Verification.** `main` and the refactor were run side by side on
SmolLM2-135M (CPU) over StrongREJECT / harmless prompts: mean activations,
per-prompt activations, mean-diff directions, refusal scores, last-token
logits, baseline vector and refusal scores under direction ablation were all
bit-identical (max |diff| = 0), and greedy generations (with and without
ablation) were identical.

## Known limitations

- **Apple GPU (MPS):** several accumulators are `float64`, which MPS does not
  support. Run on CUDA (3090 / A100) or CPU.
- **Target groups:** the shipped groups (`wmd_*`) need data that is
  intentionally not in this repo. SpeechBreak will define targets from
  categories that are in `dataset/splits` (e.g. StrongREJECT categories).
