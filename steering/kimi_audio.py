"""
Kimi-Audio-7B-Instruct wrapper: text in, text + speech out, with access to both residual streams.

  llm         alm.model.layers        28 layers, d = 3584   (Qwen2.5-7B-based; writes the reply text)
  audio_head  alm.model.mimo_layers    6 layers, d = 3584   (branches off the output of LLM layer 21;
                                                             writes the discrete audio tokens)
  detokenizer flow-matching + BigVGAN: audio tokens (12.5 Hz) -> 24 kHz waveform (not steered)

Each step the model emits one text token and one audio token in parallel; the audio stream starts
6 steps after the text stream. Both streams have the same width, so a direction from one can be
added to the other without a projection.

Needs Moonshot's inference package (kimia_infer) and flash-attn: run in the `kimi` conda env
(see README). This file does not touch the Qwen2.5-Omni code.
"""

import gc
from typing import Dict, List, Sequence, Tuple

import torch

from steering.hooks import add_hooks

MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"
STREAMS = ("llm", "audio_head")
OUTPUT_SR = 24000
AUDIO_TOKENS_PER_S = 12.5

# Moonshot's recommended sampling: greedy text, sampled audio tokens
AUDIO_TEMPERATURE, AUDIO_TOP_K = 0.8, 10


def _use_torch_vocoder_activation():
    """
    Moonshot's BigVGAN wrapper always asks for a CUDA kernel that is JIT-compiled with nvcc on first
    use; that build fails when the system CUDA toolkit doesn't match torch's. The pure-PyTorch
    activation computes the same thing (a little slower), so use it instead.
    """
    from kimia_infer.models.detokenizer import bigvgan_wrapper
    original = bigvgan_wrapper.BigVGAN
    if getattr(original, "_torch_activation", False):
        return

    class BigVGANTorchActivation(original):
        _torch_activation = True

        def __init__(self, h, use_cuda_kernel: bool = False):
            super().__init__(h, use_cuda_kernel=False)

    bigvgan_wrapper.BigVGAN = BigVGANTorchActivation


def codes_to_wav(holder, codes: List[int]) -> torch.Tensor:
    """Audio tokens -> 24 kHz waveform [samples] (cpu), with the detokenizer in holder.detokenizer."""
    from kimia_infer.api.kimia import KimiAudio
    if not codes:
        return torch.zeros(0)
    # KimiAudio.detokenize_audio only uses self.detokenizer, so it also works for KimiVocoder
    wav = KimiAudio.detokenize_audio(holder, torch.tensor(codes).unsqueeze(0))
    return wav.detach().float().cpu().view(-1)


class KimiVocoder:
    """
    Only the detokenizer (flow matching + BigVGAN, ~13 GiB at its peak), without the LLM: turns
    saved audio tokens into waveforms. Lets generation run as several LLM-only workers.
    """

    def __init__(self):
        from huggingface_hub import snapshot_download
        from kimia_infer.models.detokenizer import get_audio_detokenizer
        _use_torch_vocoder_activation()
        self.detokenizer = get_audio_detokenizer(snapshot_download(MODEL_ID))

    @torch.inference_mode()
    def __call__(self, codes: List[int]) -> torch.Tensor:
        return codes_to_wav(self, codes)


class KimiAudioModel:

    def __init__(self, load_detokenizer: bool = True, audio_input: bool = False):
        from kimia_infer.api.kimia import KimiAudio  # heavy import, kimi env only
        _use_torch_vocoder_activation()
        self.kimi = KimiAudio(model_path=MODEL_ID, load_detokenizer=load_detokenizer)
        if not audio_input:
            # Typed prompts never use the audio-input encoders (Whisper + GLM-4 voice tokenizer,
            # ~2.7 GiB): free them. Keep them (audio_input=True) to accept spoken turns.
            self.kimi.prompt_manager.audio_tokenizer = None
            self.kimi.prompt_manager.whisper_model = None
            gc.collect()
            torch.cuda.empty_cache()
        self.alm = self.kimi.alm.eval()
        self.device = next(self.alm.parameters()).device
        self.tokens = self.kimi.extra_tokens
        self.delay = self.kimi.kimia_text_audiodelaytokens  # audio stream starts this many steps late
        self.token_offset = self.kimi.kimia_token_offset     # ids >= offset are audio tokens

    # ------------------------------------------------------------------ #
    #  Architecture                                                        #
    # ------------------------------------------------------------------ #

    def blocks(self, stream: str) -> List[torch.nn.Module]:
        """Transformer blocks of the 'llm' or 'audio_head' residual stream."""
        return list(self.alm.model.layers if stream == "llm" else self.alm.model.mimo_layers)

    # ------------------------------------------------------------------ #
    #  Inputs                                                              #
    # ------------------------------------------------------------------ #

    def prompt(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (audio_ids, text_ids), each [1, T], for one typed user turn followed by the assistant start
        token: [user start] [text ...] [speech-output token] [msg end] [assistant start]. The user's
        words sit in the text stream; the audio stream holds blanks and the control tokens.

        The speech-output token (kimia_speech_ctd_id) is what Moonshot's code appends after a spoken
        user turn to ask for a spoken reply. Their text-message path leaves it out, and without it
        the speech drifts from the reply text (word error rate vs the text 1.14, against 0.22 with
        it, on 10 val prompts), so it is added here.
        """
        p = self.conversation_prompt([self.user_turn(text=text)])
        return p["audio_ids"], p["text_ids"]

    # Conversations: a list of turns (user_turn / assistant_turn), kept in memory by the caller.

    def user_turn(self, text: str = None, wav_path: str = None):
        """
        One user turn, typed (text) or spoken (wav_path; needs audio_input=True), ending with the
        speech-output token and the message end. A spoken turn carries Kimi's own audio encoding
        (GLM-4 voice tokens + Whisper features); no speech-to-text is involved. The wav file is only
        read here, so the caller can delete it afterwards.
        """
        pm = self.kimi.prompt_manager
        if wav_path is not None:
            return pm.tokenize_message({"role": "user", "message_type": "audio", "content": wav_path},
                                       tokenize_role=True, has_ct_token=True, has_msg_end_token=True,
                                       extract_whisper_feature=True, output_type="both")
        turn = pm.tokenize_message({"role": "user", "message_type": "text", "content": text},
                                   tokenize_role=True, has_ct_token=False, has_msg_end_token=False)
        for token in (self.tokens.kimia_speech_ctd_id, self.tokens.msg_end):
            turn.audio_append(token)
            turn.text_append(self.tokens.kimia_text_blank)
        return turn

    def assistant_turn(self, text: str):
        """An earlier reply of Kimi's, given back as its text."""
        return self.kimi.prompt_manager.tokenize_message(
            {"role": "assistant", "message_type": "text", "content": text},
            tokenize_role=True, has_ct_token=False, has_msg_end_token=True)

    def conversation_prompt(self, turns: Sequence) -> Dict:
        """
        Model inputs for a conversation that ends with a user turn: {"audio_ids", "text_ids"} [1, T],
        plus "mask" / "whisper" (None unless a turn was spoken) for the positions holding audio.
        """
        from kimia_infer.utils.data import KimiAContent
        content = KimiAContent()
        for turn in turns:
            content.merge(turn)
        content.merge(self.kimi.prompt_manager.tokenize_message(
            {"role": "assistant", "message_type": None}, tokenize_role=True,
            has_ct_token=False, has_msg_end_token=False))
        audio_ids, text_ids, mask, _, _ = content.to_tensor()
        spoken = bool(content.continuous_feature)
        return {"audio_ids": audio_ids.to(self.device), "text_ids": text_ids.to(self.device),
                "mask": mask.to(self.device) if spoken else None,
                "whisper": [f.to(self.device) for f in content.continuous_feature] if spoken else None}

    def continuation(self, text_tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """(audio_ids, text_ids) for reply text tokens with a blank audio stream (text-only steps)."""
        text_ids = torch.tensor([text_tokens], dtype=torch.long, device=self.device)
        return torch.full_like(text_ids, self.tokens.kimia_text_blank), text_ids

    def encode_text(self, text: str) -> List[int]:
        return self.kimi.prompt_manager.text_tokenizer.encode(text, bos=False, eos=False)

    # ------------------------------------------------------------------ #
    #  Forward / generation                                                #
    # ------------------------------------------------------------------ #

    def forward(self, audio_ids: torch.Tensor, text_ids: torch.Tensor, past=None, start: int = 0,
                whisper=None, mask=None):
        """
        One pass over [1, T] tokens -> (audio_logits, text_logits, past_key_values). whisper / mask:
        the Whisper features and positions of spoken user turns (prompt pass only).
        """
        position_ids = torch.arange(start, start + audio_ids.shape[1], device=self.device).unsqueeze(0)
        return self.alm.forward(input_ids=audio_ids, text_input_ids=text_ids, whisper_input_feature=whisper,
                                is_continuous_mask=mask, position_ids=position_ids,
                                past_key_values=past, return_dict=False)[:3]

    @torch.inference_mode()
    def greedy_text(self, text: str, n: int) -> List[int]:
        """First n reply text tokens, greedy, audio stream blank."""
        audio_ids, text_ids = self.prompt(text)
        pos, past, out = 0, None, []
        for _ in range(n):
            _, text_logits, past = self.forward(audio_ids, text_ids, past, pos)
            pos += audio_ids.shape[1]
            out.append(int(text_logits[0, -1].argmax()))
            audio_ids, text_ids = self.continuation(out[-1:])
        return out

    @torch.inference_mode()
    def generate(self, text: str = None, max_seconds: float = 30.0, prompt: Dict = None) -> Dict:
        """
        Reply to a typed prompt: {"text": str, "codes": audio tokens (12.5 per second), "audio":
        24 kHz waveform tensor, or None if the detokenizer isn't loaded (see KimiVocoder)}.
        Same loop as Moonshot's KimiAudio._generate_loop (greedy text, sampled audio tokens), capped
        at max_seconds of speech. Steering hooks registered by the caller apply at every step.
        prompt: a conversation_prompt(...) to answer instead of the single typed turn `text`.
        """
        from kimia_infer.utils.sampler import KimiASampler
        sampler = KimiASampler(audio_top_k=AUDIO_TOP_K, audio_temperature=AUDIO_TEMPERATURE,
                               audio_repetition_penalty=1.0, audio_repetition_window_size=64,
                               text_top_k=5, text_temperature=0.0,
                               text_repetition_penalty=1.0, text_repetition_window_size=16)
        prompt = prompt or self.conversation_prompt([self.user_turn(text=text)])
        audio_ids, text_ids = prompt["audio_ids"], prompt["text_ids"]
        whisper, mask = prompt["whisper"], prompt["mask"]
        max_steps = self.delay + int(max_seconds * AUDIO_TOKENS_PER_S)
        blank, eod = self.tokens.kimia_text_blank, (self.tokens.msg_end, self.tokens.media_end)

        pos, past, text_done = 0, None, False
        text_tokens, audio_tokens = [], []
        for i in range(max_steps):
            audio_logits, text_logits, past = self.forward(audio_ids, text_ids, past, pos, whisper, mask)
            whisper = mask = None  # only the prompt holds spoken audio
            pos += audio_ids.shape[1]
            next_text = sampler.sample_text_logits(text_logits)
            next_audio = sampler.sample_audio_logits(audio_logits)

            if text_done:
                next_text.fill_(blank)
            elif next_text.item() == self.tokens.kimia_text_eos:
                text_done = True
            else:
                text_tokens.append(next_text.item())

            if i < self.delay:
                next_audio.fill_(blank)
            elif next_audio.item() in eod:
                break
            else:
                audio_tokens.append(next_audio.item())
            audio_ids, text_ids = next_audio.unsqueeze(1), next_text.unsqueeze(1)

        reply = self.kimi.prompt_manager.text_tokenizer.decode([t for t in text_tokens if t < self.token_offset])
        codes = [t - self.token_offset for t in audio_tokens if t >= self.token_offset]
        wav = None if self.kimi.detokenizer is None else codes_to_wav(self.kimi, codes)
        return {"text": reply, "codes": codes, "audio": wav}

    # ------------------------------------------------------------------ #
    #  Activations                                                         #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def extract_activations(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Residual stream (block input) at the last prompt token - the position that predicts the
        first reply token - for every layer of both streams, from one forward pass per prompt:
        {"llm": [n, 28, d], "audio_head": [n, 6, d]} (float32, cpu).
        """
        store = {}

        def make_hook(stream, layer):
            def hook_fn(module, args):
                store[(stream, layer)] = args[0][0, -1, :].detach().float().cpu()
            return hook_fn

        hooks = [(b, make_hook(s, l)) for s in STREAMS for l, b in enumerate(self.blocks(s))]
        out = {s: [] for s in STREAMS}
        for text in texts:
            store.clear()
            audio_ids, text_ids = self.prompt(text)
            with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
                self.forward(audio_ids, text_ids)
            for s in STREAMS:
                out[s].append(torch.stack([store[(s, l)] for l in range(len(self.blocks(s)))]))
        return {s: torch.stack(v) for s, v in out.items()}

    def unload(self):
        del self.alm, self.kimi
        gc.collect()
        torch.cuda.empty_cache()
