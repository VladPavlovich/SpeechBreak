# SpeechBreak: steering refusal in Qwen2.5-Omni through its residual stream

Qwen2.5-Omni-7B listens to a spoken request and answers out loud. It has two transformers:
the **Thinker** (28 layers) understands the audio and writes the reply text, and the **Talker**
(24 layers) turns that text into speech.

This repo finds a **harmful-vs-harmless direction** in those residual streams, picks the layers where
it matters, and steers Qwen with it: subtracting the direction makes Qwen answer requests it
would refuse (jailbreak), and adding it makes Qwen refuse ordinary ones.

## Findings

The best settings and the evidence behind them are in [results/best_settings.json](results/best_settings.json).

| Goal | Setting | Result |
|---|---|---|
| **Jailbreak** | Thinker layer 18 or 19, α = −0.5 to −1 | The one harmful val request Qwen refused unsteered (a "vaccines cause autism" Twitter thread) was answered under 8 of the 9 Thinker settings tried (layer 20 at α = −0.5 only half-complied); no broken replies |
| **Refuse** | Thinker layer 18, α = +2 | 5/5 harmless requests refused (0/5 unsteered), with nonsense reasons ("CSS … illegal activities") |

- **Steer the Thinker.** Steering the Talker as well adds nothing.
- **The Talker only changes the voice.** Talker steering never changes what Qwen says; at α = 2 it
  breaks the speech (stuttering).
- **Cross-transfer barely works.** The Talker direction mapped into the Thinker has no effect; the
  Thinker direction mapped into the Talker acts like noise.

α is in raw direction units: α = 1 moves an average harmless activation onto an average harmful one.

**Caveats.** Each setting was tested on 5 prompts. The judge mislabels some unsteered harmful
replies (it calls produced harassment "RESIST"), so the jailbreak numbers in
`results/5_evaluation/summary.md` look stronger than they are: unsteered, Qwen refused only 1 of the
5 harmful val prompts. The test split hasn't been used yet.

## Repo layout

```
dataset/                 spoken requests (Kokoro TTS) and the scripts that made them
  audio/harmful_audio/     155 HarmBench requests: illegal, misinformation, harassment
  audio/harmless_audio/    985 Alpaca instructions
steering/                the code
  qwen_omni.py             Qwen2.5-Omni wrapper: audio in, speech out, activations of both streams
  data.py                  loads the spoken requests and the frozen train / val / test split
  hooks.py                 adds a steering vector to a layer's input
  scoring.py               Whisper transcripts + Qwen2.5-7B-Instruct judge
  run_steps.py             the six steps below
scripts/
  steer_prompts.py         steer Qwen on your own prompts and save the spoken replies
  steer_prompts.txt        your prompts, one per line
  run_steps.sh             runs the six steps (parallel GPU workers for step 4)
  demo_prompt.txt          the prompt step 6 speaks
  tts_batch.py             Kokoro text-to-speech helper (runs in the "tts" conda env)
results/
  best_settings.json       layers and alphas found, best setting per experiment, findings
  splits.json              frozen train / val / test split (wav file names)
  1_activations/ … 6_demo/ outputs of each step
  steer_prompts/           replies generated with scripts/steer_prompts.py
```

## Setup

```bash
pip install -r requirements.txt                          # main env: Qwen2.5-Omni, Whisper, judge
conda create -n tts python=3.11
conda run -n tts pip install kokoro==0.9.4 soundfile     # text-to-speech env
```

Qwen2.5-Omni-7B needs about 25 GB of GPU memory. Everything here ran on one A100 80GB.

## Steer Qwen on your own prompts

Write prompts in [scripts/steer_prompts.txt](scripts/steer_prompts.txt), one per line, then:

```bash
python scripts/steer_prompts.py                                  # the 4 jailbreak settings
python scripts/steer_prompts.py --settings L18_a-1 baseline      # one setting + an unsteered reply
python scripts/steer_prompts.py --custom 20:-1 18:-2             # any Thinker layer:alpha
python scripts/steer_prompts.py --asr                            # also transcribe the replies
```

Each prompt is spoken with Kokoro and answered once per setting, with the same random seed, so the
steering is the only difference. Replies go to `results/steer_prompts/<timestamp>/`
(`01_input.wav`, `01_L18_a-1.wav`, …, `results.json` with Qwen's text). About 40 s per reply.

Settings: `L18_a-0.5`, `L18_a-1`, `L19_a-0.5`, `L19_a-1` (jailbreak, the default),
`refuse_L18_a+2`, `baseline`.

## Reproduce the search

```bash
nohup bash scripts/run_steps.sh > run_steps.out 2>&1 &
```

| Step | What it does | Output |
|---|---|---|
| 1 extract | Residual stream at the last audio token of every train / val clip, both streams | `results/1_activations/` |
| 2 directions | Per layer: mean(harmful) − mean(harmless) on train; how well it separates val clips | `results/2_directions/` |
| 3 sweep | Adds the direction at each layer and measures the shift in Qwen's first reply token, forward passes only (no generation); keeps the top layers | `results/3_layer_sweep/layer_table_*.md` |
| 4 generate | Spoken replies at the top layers × α ∈ {0.5, 1, 2} × 5 prompts, for five experiments: Thinker only, Talker only, both, Thinker→Talker, Talker→Thinker | `results/4_generations/` |
| 5 evaluate | Whisper transcript + judge (RESIST / COMPLY / NEUTRAL / INCOHERENT) of every reply; best setting per experiment | `results/5_evaluation/summary.md` |
| 6 demo | Speaks `scripts/demo_prompt.txt` and answers it with the best setting | `results/6_demo/demo.json` |

Steps resume where they stopped. Run a subset with `STEPS="generate evaluate"`; see the top of
[scripts/run_steps.sh](scripts/run_steps.sh) for all options. Step 4 (≈370 replies) is the slow one:
about 1.5 hours with 3 parallel workers on an A100.

The layer search follows the refusal-direction method of
[Arditi et al. (2024)](https://github.com/andyrdt/refusal_direction): a difference-in-means
direction, with layers chosen from forward passes rather than generations.
