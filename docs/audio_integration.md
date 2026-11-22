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
| 2 | Audio model adapters: Qwen2-Audio | Done (structural tests pass; real-weights run pending on GPU) |
| 2b | More adapters: Ultravox, Qwen2.5-Omni | Planned |
| 3 | Voice data: TTS of existing splits, VoiceBench loader, `--modality` / `--train_modality` | Done (tested on CPU with tiny model) |
| 4 | Judging: unchanged (judges read transcript + response) | — |
| 5 | Gradio UI (`app.py`): run pipeline, browse results, try it with mic/upload | Done (tested in browser with tiny model + synthetic run) |

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

## Phase 2: Qwen2-Audio adapter

`pipeline/model_utils/qwen2_audio_model.py` (`Qwen2AudioModel`). The factory
routes any model path containing `qwen2-audio` / `qwen2_audio` to it (checked
before the generic `qwen` match):

```bash
python3 -m pipeline.run_pipeline_repit --model_path Qwen/Qwen2-Audio-7B-Instruct --target_group <group>
```

**Data format.** Items are the usual dicts with an optional `audio` field:

```python
{"instruction": "transcript (used by judges and logs)", "category": "...", "audio": <audio>}
```

`<audio>` is anything `pipeline/utils/audio_utils.load_audio` accepts: a file
path, a HuggingFace audio dict (`{"array", "sampling_rate"}` or `{"path"}`),
a numpy array or a torch tensor. It is converted to 16 kHz mono. Items without
`audio` (or plain strings) are text-only prompts, so text and spoken prompts
can be mixed in a batch.

**Prompt layout.**

```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|>×N<|audio_eos|>\n[text]<|im_end|>\n<|im_start|>assistant\n
                                                                  └──────── eoi tokens (hooked) ───────┘
```

- `audio_text_mode="audio_only"` (default): spoken prompt only; the
  transcript is *not* shown to the model.
- `audio_text_mode="append"`: audio followed by the transcript as text.
- `system=None` (what the pipeline passes) means the default system prompt;
  pass `system=""` to omit it.
- Hooks go on the Qwen2 decoder layers (`model.language_model.model.layers`);
  the audio encoder is left untouched.

**Pipeline fix found while testing.** `_group_prompts_by_category`
(`generate_directions_repit.py`) kept only `obj["instruction"]`, which would
have silently computed category directions from transcripts. It now keeps the
whole item when it has audio (text items are unchanged).

### Tests

`tests/test_qwen2_audio_adapter.py` runs the real pipeline functions on
text-only, audio-only and mixed batches and checks: factory routing, processor
keys, eoi tokens at the end of every left-padded row, decoder hooks seeing the
full sequence, audio changing activations, left-padding invariance, category
directions using audio, refusal scores / baseline / logits, ablation hooks
changing scores and generations, CE-loss masking of audio tokens, and
`audio_text_mode="append"`.

Fast CPU check with a tiny random model (only the processor files are
downloaded, a few MB):

```bash
python -m tests.make_tiny_qwen2_audio /tmp/tiny-qwen2-audio
TOKENIZERS_PARALLELISM=false python -m tests.test_qwen2_audio_adapter /tmp/tiny-qwen2-audio --device cpu
```

Real model on a GPU (weights ~16 GB in bf16; fits a 24 GB 3090). This also
prints a few real answers as a sanity check:

```bash
TOKENIZERS_PARALLELISM=false python -m tests.test_qwen2_audio_adapter Qwen/Qwen2-Audio-7B-Instruct
```

**Hardware.** A 3090 (24 GB) is enough for the adapter test and for direction
extraction with small batches. Full pipeline runs (7B model plus
LlamaGuard-3-8B judging, long generations) are more comfortable on an A100.
Lower the pipeline's batch sizes (`batch_size_denom` in `run_pipeline`) on a
3090: audio prompts are much longer than text ones (about 25 tokens per
second of audio).

## Phase 3: voice data and modality flags

### Running

```bash
# spoken prompts everywhere
python3 -m pipeline.run_pipeline_repit --model_path Qwen/Qwen2-Audio-7B-Instruct --target_group hate --modality audio

# cross-modal transfer: directions from text prompts, tested on spoken prompts
python3 -m pipeline.run_pipeline_repit --model_path Qwen/Qwen2-Audio-7B-Instruct --target_group hate \
    --train_modality text --modality audio

# text baseline on the same audio model
python3 -m pipeline.run_pipeline_repit --model_path Qwen/Qwen2-Audio-7B-Instruct --target_group hate --modality text
```

| Flag | Values | Meaning |
|---|---|---|
| `--modality` | `text` (default), `audio`, `audio+text` | What the model sees for held-out test prompts and eval datasets. `audio+text` = speech followed by its transcript. |
| `--train_modality` | same, defaults to `--modality` | What the model sees for direction extraction and selection (train/val). `audio` and `audio+text` can't be mixed in one run. |
| `--tts` | `mms` (default), `mms:<hf id>`, `say`, `say:<voice>` | How text prompts are turned into speech. |

Runs are saved under `pipeline/runs/<model>-RepIt-<group>[-<modality>][-from_<train_modality>]`,
so text and audio runs of the same model don't overwrite each other.

### Target groups

The `wmd_*` groups need data that isn't in this repo. These groups use
categories that already exist in the shipped jailbreakv / strongreject splits:

| Group | Categories |
|---|---|
| `hate` | Hate_Speech, Hate,_harassment_and_discrimination, Bias |
| `fraud` | Fraud, Economic_Harm |
| `privacy` | Privacy_Violation |
| `disinfo` | Disinformation_and_deception |

### Where the audio comes from (`dataset/audio_data.py`)

- **TTS, on demand and cached.** When a modality isn't `text`, the loader
  speaks each sampled prompt and caches the WAV under `dataset/audio/cache/`
  (git-ignored). The cache is keyed by backend, voice and text, so reruns and
  other target groups reuse clips. Set `SPEECHBREAK_AUDIO_CACHE` to put it on a
  bigger disk. The default backend is `facebook/mms-tts-eng` (VITS, ~150 MB,
  16 kHz, deterministic per prompt). It's a single neutral voice; voice
  variety (Kokoro, VoiceBench's `alpacaeval_speaker` voices) is a later step.
- **VoiceBench.** `voicebench_advbench` (520 spoken AdvBench prompts) and
  `voicebench_alpacaeval` (spoken harmless prompts) load through
  `load_dataset(...)`. Audio runs add `voicebench_advbench` to the eval sets
  automatically; under `--modality text` it falls back to its transcripts.
- **Long prompts.** The audio encoder sees at most 30 s. The loader warns
  when prompts are likely longer than that; they get truncated.

Pre-warming the cache (optional, e.g. on a CPU node before a GPU run):

```bash
python -m dataset.audio_data --tts mms --splits strongreject_harmful_train jailbreakv_harmful_train harmless_train
python -m dataset.audio_data --tts mms --datasets advbench jailbreakbench harmbench tdc2023 malicious_instruct
```

Note `harmless_train` has ~18.8k prompts; the pipeline only samples as many
as there are harmful ones, so lazy synthesis during the run does far less work.

### Tests

```bash
TOKENIZERS_PARALLELISM=false python -m tests.test_voice_data /tmp/tiny-qwen2-audio --device cpu
```

Checks TTS output and caching, `apply_modality`, the VoiceBench loader,
config validation, the real dataset loader with `train=text, test=audio`,
and audio items going through direction extraction, layer selection,
candidate isolation and generation (including a VoiceBench batch).

## Phase 5: UI (`app.py`)

```bash
python app.py            # http://127.0.0.1:7860
```

On the GPU server, keep the default `127.0.0.1` bind and tunnel from your
laptop instead of exposing the port or using a public Gradio share link (the
Try-it tab can produce unsafe outputs):

```bash
ssh -L 7860:localhost:7860 <user>@<gpu-server>   # then open http://localhost:7860
```

**Run.** Choose the model, target group, modality, train modality, TTS
backend, GPU (`CUDA_VISIBLE_DEVICES`) and study flags. The exact command is
shown before you start. The pipeline runs as a background process (one at a
time) with a live log; Stop kills the whole process group. Logs are kept in
`pipeline/runs/_ui_logs/`, and runs keep going if you close the browser.

**Results.** Pick a run to see:
- attack success rate per intervention (Baseline, Nontarget, Natural, RepIt,
  Projection) for each judge (LlamaGuard3, substring), split into target and
  non-target categories. RepIt working means high target ASR with low
  non-target ASR;
- per-category ASR on the held-out harmful split;
- the chosen layer, position and rho, plus every plot the run saved;
- a completions browser with search and a "judged jailbreaks only" filter.

**Try it.** Load a model (~16 GB of GPU memory for the 7B; don't share a GPU
with a running pipeline), pick a finished run and either its RepIt direction
(target concept only) or the full refusal direction (rho = 0), then record,
upload or type a prompt. You get the baseline answer and the answer with the
direction ablated side by side.

`SPEECHBREAK_RUNS_DIR` points the UI at a different runs folder.

## Known limitations

- **Apple GPU (MPS):** several accumulators are `float64`, which MPS does not
  support. Run on CUDA (3090 / A100) or CPU.
- **Target groups:** the shipped groups (`wmd_*`) need data that is
  intentionally not in this repo. SpeechBreak will define targets from
  categories that are in `dataset/splits` (e.g. StrongREJECT categories).
