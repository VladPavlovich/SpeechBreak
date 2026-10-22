# SpeechBreak: steering refusal in speech models (Qwen2.5-Omni and Kimi-Audio)

Qwen2.5-Omni-7B listens to a spoken request and answers out loud. It has two transformers:
the **Thinker** (28 layers) understands the audio and writes the reply text, and the **Talker**
(24 layers) turns that text into speech.

This repo finds a **harmful-vs-harmless direction** in those residual streams, picks the layers where
it matters, and steers Qwen with it: subtracting the direction makes Qwen answer requests it
would refuse (jailbreak), and adding it makes Qwen refuse ordinary ones.

The same experiments are also run on a second model, **Kimi-Audio-7B-Instruct**: see
[Second model: Kimi-Audio](#second-model-kimi-audio-7b-instruct). The Qwen and Kimi code and results
are separate. Both models can also be steered live from the browser, by voice or text: see
[Live demo](#live-demo-talk-to-the-models-in-your-browser).

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
dataset/                 the spoken requests the directions are computed from
  audio/harmful_audio/     155 HarmBench requests: illegal, misinformation, harassment (+ manifest.json)
  audio/harmless_audio/    985 Alpaca instructions (+ manifest.json)
  processed/harmbench_illegal_harassment_misinfo.json, processed/harmless_alpaca.json   their text
  generate_audio.py        text -> the two audio folders (Kokoro TTS)
steering/                the code
  qwen_omni.py             Qwen2.5-Omni wrapper: audio in, speech out, activations of both streams
  data.py                  loads the spoken requests and the frozen train / val / test split
  hooks.py                 adds a steering vector to a layer's input
  scoring.py               Whisper transcripts + Qwen2.5-7B-Instruct judge
  run_steps.py             the six steps below
  kimi_audio.py            Kimi-Audio wrapper: text or speech in, text + speech out, activations of both streams
  run_steps_kimi.py        the same six steps for Kimi-Audio
scripts/
  steer_prompts.py         steer Qwen on your own prompts and save the spoken replies
  steer_prompts.txt        your prompts, one per line
  run_steps.sh             runs the six steps (parallel GPU workers for step 4)
  demo_prompt.txt          the prompt step 6 speaks
  tts_batch.py             Kokoro text-to-speech helper (runs in the "tts" conda env)
  live.sh, live_qwen.py, live_kimi.py, live_common.py   talk to either model in your browser
  steer_prompts_kimi.py    steer Kimi-Audio on your own prompts
  steer_prompts_kimi.txt   your prompts for Kimi-Audio, one per line
  run_steps_kimi.sh        runs the six steps for Kimi-Audio
results/
  best_settings.json       layers and alphas found, best setting per experiment, findings
  splits.json              frozen train / val / test split (wav file names)
  1_activations/ … 6_demo/ outputs of each step
  steer_prompts/           replies generated with scripts/steer_prompts.py
results_kimi/            the same layout for Kimi-Audio
```

## Setup

```bash
pip install -r requirements.txt                          # main env: Qwen2.5-Omni, Whisper, judge
conda create -n tts python=3.11
conda run -n tts pip install kokoro==0.9.4 soundfile     # text-to-speech env
```

Qwen2.5-Omni-7B needs about 25 GB of GPU memory. Everything here ran on one A100 80GB.

The spoken requests live in `dataset/audio/` (not in git, ~400 MB). To regenerate them from
the text (entry i uses Kokoro voice i mod 14; existing wavs are skipped):

```bash
conda run -n tts --no-capture-output python dataset/generate_audio.py
```

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

## Second model: Kimi-Audio-7B-Instruct

[Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) is one 28-layer LLM that writes the reply
text, plus a 6-layer **audio head** that branches off LLM layer 21 and writes the audio tokens; a
separate detokenizer turns those into a waveform. The six steps are the same as for Qwen, with
these differences:

| | Qwen2.5-Omni | Kimi-Audio |
|---|---|---|
| Input | spoken request (`dataset/audio/`) | the same requests as typed text (`dataset/processed/*.json`), no TTS |
| Streams | Thinker (28) → Talker (24), different widths | LLM (28) and audio head (6), same width, so cross-stream directions need no projection |
| Direction read at | last audio token | last prompt token |
| Split | `results/splits.json` | the same items |
| Results | `results/` | `results_kimi/` |

Typed prompts are followed by Kimi's speech-output control token. Moonshot's code only adds it
after spoken prompts; without it Kimi's speech drifts away from its own reply text (see
`KimiAudioModel.prompt` in [steering/kimi_audio.py](steering/kimi_audio.py)).

### Findings (Kimi)

Details in [results_kimi/best_settings.json](results_kimi/best_settings.json).

| Goal | Setting | Result |
|---|---|---|
| **Jailbreak** | LLM layer 19, α = −1 | All 5 harmful val prompts answered, in text and intelligible speech, including the two Kimi declined or rebutted unsteered |
| **Refuse** | LLM layer 19, α = +2 | The reply text refuses 5/5 harmless prompts, but the speech turns to gibberish; at α = +1 the speech stays intelligible and 2/5 are refused |

- **The speech breaks before the text does.** The audio head reads LLM layer 21, so steering the
  LLM at or below it also disturbs the speech: fine at α ≤ 1, garbled at α = 2.
- **Layer 19 beats 20 and 21**, which often collapse into "Hey there!" loops.
- **Steering only the audio head doesn't change the words.**
- **The prompt mode matters.** With the speech-output token Kimi answers 3 of the 5 harmful val
  prompts unsteered; in plain text mode it refused 4 of the same 5.

The same caveats apply: 5 prompts per setting, a judge that mislabels some replies (these findings
come from reading the transcripts), and an unused test split. Kimi also sometimes answers English
prompts in Chinese in this mode, even unsteered.

### Setup (Kimi)

Kimi-Audio needs its own conda env: its model code doesn't load with newer `transformers`, and its
attention requires `flash-attn`.

```bash
conda create -n kimi python=3.10 -y
conda run -n kimi pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
conda run -n kimi pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
git clone --recursive https://github.com/MoonshotAI/Kimi-Audio /tmp/Kimi-Audio
conda run -n kimi pip install --no-deps /tmp/Kimi-Audio
conda run -n kimi pip install transformers==4.51.3 diffusers==0.33.1 ninja tiktoken loguru librosa soundfile \
    conformer==0.3.2 torchdyn==1.0.6 timm hyperpyyaml omegaconf==2.3.0 openai-whisper easydict sentencepiece \
    "gradio<6" huggingface_hub==0.36.2      # gradio: only for the live demo
```

The weights (43 GB) download from Hugging Face on first use. The BigVGAN vocoder's CUDA kernel is
replaced by its PyTorch version at load time, so no CUDA compiler is needed. Scoring (Whisper + judge) runs in the
main env, as for Qwen; the runner switches envs per step.

### Run (Kimi)

```bash
nohup bash scripts/run_steps_kimi.sh > run_steps_kimi.out 2>&1 &               # all six steps, ~1 hour
conda run -n kimi --no-capture-output python scripts/steer_prompts_kimi.py     # your prompts, jailbreak settings
conda run -n kimi --no-capture-output python scripts/steer_prompts_kimi.py --settings L19_a-1 baseline
conda run -n kimi --no-capture-output python scripts/steer_prompts_kimi.py --custom 18:-1 19:-1.5
```

`steer_prompts_kimi.py` reads [scripts/steer_prompts_kimi.txt](scripts/steer_prompts_kimi.txt) (one
prompt per line) and writes to `results_kimi/steer_prompts/<timestamp>/`. Settings: `L19_a-1`,
`L19_a-0.5` (jailbreak, the default), `refuse_L19_a+1`, `refuse_L19_a+2`, `baseline`.

Step 4 fills the GPU in two passes: 3 workers that load only the LLM (about 20 GB each) write the
reply text and audio tokens, then 5 workers that load only the detokenizer (about 13 GB each) turn
the tokens into wavs.

## Live demo: talk to the models in your browser

```bash
bash scripts/live.sh qwen     # Qwen2.5-Omni on http://localhost:7860   (~25 GB of GPU memory)
bash scripts/live.sh kimi     # Kimi-Audio   on http://localhost:7861   (~33 GB)
bash scripts/live.sh          # both
```

The pages are served on the GPU server's own `127.0.0.1`. From your laptop, tunnel the ports and
open the URLs above (browsers only allow the microphone on `localhost` or https):

```bash
ssh -L 7860:localhost:7860 -L 7861:localhost:7861 <you>@<gpu-server>
```

On each page you speak into the microphone or type, and the model answers out loud. Your speech or
text goes straight into the model and the model's own audio comes back: there is no text-to-speech
or speech-to-text step. A menu switches between unsteered, the best settings from
`best_settings.json`, and any stream / layer / α; it can be changed between turns. The conversation
continues over turns until you press *New conversation*.

Nothing is saved: replies are sent to the browser as inline audio and never written to disk, a
microphone recording is deleted as soon as the model has read it, and the conversation is held in
memory per browser tab and dropped when the tab closes ([scripts/live_common.py](scripts/live_common.py)).

Speed: a reply takes roughly twice its own length when the GPU is shared with another job (8 s of
speech in ~18 s for Qwen); the *Max reply length* slider (default 15 s) is the main lever.

Tested so far: the Qwen page end to end (typed and spoken turns, multi-turn memory, steering, no files
left behind). The Kimi page has only been checked up to building the prompt (typed, spoken and
multi-turn conversations assemble correctly); its spoken input and multi-turn replies have not been
run on the GPU yet. Kimi's steering directions were found on typed prompts and are applied the same
way to spoken ones.
