"""
Shared front end of the live demos (scripts/live_qwen.py, scripts/live_kimi.py).

One page: speak or type, the model answers out loud; pick the steering; the conversation continues
over turns until "New conversation".

Nothing is kept once the page is closed:
  - replies are sent to the browser as inline audio and are never written to disk
  - a microphone recording is deleted as soon as the model has read it
  - the conversation lives in this process's memory, per browser tab, and is dropped when the tab
    closes (and when you press "New conversation")
  - Gradio's temp folder is a private directory that is emptied every minute and removed on exit
"""

import atexit
import base64
import io
import os
import shutil
import signal
import sys
import tempfile
import time

# Must be set before gradio is imported: uploads (microphone recordings) go to a private folder
PRIVATE_TMP = tempfile.mkdtemp(prefix="speechbreak-live-")
os.environ["GRADIO_TEMP_DIR"] = PRIVATE_TMP
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
atexit.register(shutil.rmtree, PRIVATE_TMP, ignore_errors=True)
# `kill` (SIGTERM) skips atexit: turn it into a normal exit so the folder is removed then too
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

SAMPLE_RATE = 24000
UNSTEERED, CUSTOM = "Unsteered", "Custom (pick stream, layer, α)"
SESSIONS = {}  # browser tab (session hash) -> list of conversation turns, in memory only


def audio_html(wav: np.ndarray) -> str:
    """The reply as an inline (base64) audio player that starts playing; no file on the server."""
    if wav is None or not len(wav):
        return "<i>(no audio in this reply)</i>"
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<audio controls autoplay style="width:100%" src="data:audio/wav;base64,{b64}"></audio>'


def on_gpu(fn, *args):
    """Run a generation; if the GPU runs out of memory, say so on the page and keep the app alive."""
    try:
        return fn(*args)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        raise gr.Error(f"Out of GPU memory ({free:.1f} GiB free; other jobs may be using the GPU). "
                       "Try a shorter max reply length, or free the GPU and retry.")


def build_ui(backend):
    """
    backend:
      title, intro           page heading and one paragraph
      presets                {label: (layer, signed alpha)} from the model's best_settings.json
      streams                {label: stream name} for the custom setting
      max_layer, default_layer
      preset_spec(layer)     steering spec for a preset layer
      user_turn(mic_path, text), assistant_turn(text)   conversation turns (kept in memory)
      reply(turns, spec, alpha, max_seconds) -> (24 kHz waveform, reply text)
    """
    choices = [UNSTEERED] + list(backend.presets) + [CUSTOM]

    def steering_choice(preset, stream_label, layer, alpha):
        if preset == UNSTEERED:
            return None, 0.0, "unsteered"
        if preset == CUSTOM:
            stream = backend.streams[stream_label]
            return [[stream, int(layer), stream, int(layer)]], float(alpha), f"{stream} layer {int(layer)}, α = {alpha:+g}"
        layer, alpha = backend.presets[preset]
        return backend.preset_spec(layer), alpha, preset

    def respond(mic, text, preset, stream_label, layer, alpha, max_seconds, chat, request: gr.Request):
        text = (text or "").strip()
        try:
            if not mic and not text:
                raise gr.Error("Record something with the microphone, or type a message.")
            user = backend.user_turn(mic, text)  # a recording is read into memory here
        finally:
            if mic and os.path.exists(mic):
                os.remove(mic)
        turns = SESSIONS.setdefault(request.session_hash, [])
        spec, signed_alpha, label = steering_choice(preset, stream_label, layer, alpha)

        t0 = time.time()
        wav, reply = on_gpu(backend.reply, turns + [user], spec, signed_alpha, max_seconds)
        turns += [user, backend.assistant_turn(reply)]
        chat = chat + [
            {"role": "user", "content": "🎤 (spoken)" if mic else text},
            {"role": "assistant", "content": f"{reply}\n\n*({label}; {len(wav) / SAMPLE_RATE:.0f} s of speech "
                                             f"in {time.time() - t0:.0f} s)*"},
        ]
        return audio_html(wav), chat, None, ""

    def new_conversation(request: gr.Request):
        SESSIONS.pop(request.session_hash, None)
        return "", [], None, ""

    def forget(request: gr.Request):  # the tab was closed or reloaded
        SESSIONS.pop(request.session_hash, None)

    def toggle_custom(preset):
        show = preset == CUSTOM
        return gr.update(visible=show), gr.update(visible=show), gr.update(visible=show)

    with gr.Blocks(title=backend.title, delete_cache=(60, 60), analytics_enabled=False) as demo:
        gr.Markdown(f"# {backend.title}\n{backend.intro}\n\n"
                    "Steering adds α × the harmful-vs-harmless direction to one layer: **negative α = jailbreak** "
                    "(fewer refusals), **positive α = refuse more**. You can change it between turns. "
                    "Nothing is saved: replies exist only in this tab and the conversation is forgotten when "
                    "you close it.")
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(choices, value=UNSTEERED, label="Steering")
                stream = gr.Radio(list(backend.streams), value=list(backend.streams)[0], label="Stream", visible=False)
                layer = gr.Slider(0, backend.max_layer, value=backend.default_layer, step=1, label="Layer", visible=False)
                alpha = gr.Slider(-3, 3, value=-1, step=0.25, label="α", visible=False)
                max_seconds = gr.Slider(5, 60, value=15, step=5, label="Max reply length (seconds; shorter = faster)")
                reset = gr.Button("New conversation")
            with gr.Column(scale=2):
                chat = gr.Chatbot(type="messages", height=340, label="Conversation (text of the spoken replies)")
                reply_audio = gr.HTML(label="Reply")
                mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
                text = gr.Textbox(label="…or type", placeholder="Type a message and press Enter")
                send = gr.Button("Send", variant="primary")

        preset.change(toggle_custom, preset, [stream, layer, alpha])
        inputs = [mic, text, preset, stream, layer, alpha, max_seconds, chat]
        outputs = [reply_audio, chat, mic, text]
        send.click(respond, inputs, outputs)
        text.submit(respond, inputs, outputs)
        mic.stop_recording(respond, inputs, outputs)  # answer as soon as you stop talking
        reset.click(new_conversation, None, outputs)
        demo.unload(forget)
    return demo


def launch(backend, host: str, port: int, share: bool):
    print(f"Ready: open http://localhost:{port} (via an SSH tunnel if the GPU server is remote)", flush=True)
    build_ui(backend).queue(default_concurrency_limit=1).launch(
        server_name=host, server_port=port, share=share, show_error=True)
