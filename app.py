"""SpeechBreak UI.

    python app.py                      # http://127.0.0.1:7860
    python app.py --port 7861

On a GPU server, keep the default 127.0.0.1 bind and tunnel from your laptop:

    ssh -L 7860:localhost:7860 <server>     # then open http://localhost:7860

Tabs:
  Run      launch the RepIt pipeline (text or spoken prompts), stream its log, stop it
  Results  per-run ASR tables (target / non-target / overall), plots, completions
  Try it   speak or type a prompt; compare the model with and without a run's direction
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.realpath(__file__))
RUNS_DIR = os.environ.get("SPEECHBREAK_RUNS_DIR", os.path.join(ROOT, "pipeline", "runs"))
LOG_DIR = os.path.join(RUNS_DIR, "_ui_logs")

DEFAULT_MODELS = ["Qwen/Qwen2-Audio-7B-Instruct"]
MODALITIES = ["text", "audio", "audio+text"]
INTERVENTIONS = ["Baseline", "Nontarget", "Natural", "RepIt", "Projection"]


def target_groups() -> List[str]:
    from pipeline.run_pipeline_repit import BUILTIN_TARGET_GROUPS
    return list(BUILTIN_TARGET_GROUPS)


# =============================================================================
# Run tab: one pipeline subprocess at a time
# =============================================================================
class RunManager:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: Optional[str] = None
        self.cmd: List[str] = []
        self.started: Optional[float] = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: List[str], env_extra: Dict[str, str]) -> str:
        if self.running():
            raise gr.Error("A run is already in progress. Stop it first.")
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", **env_extra}
        log = open(self.log_path, "w")
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        # own process group so Stop also kills child workers
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.cmd, self.started = cmd, time.time()
        return self.log_path

    def stop(self) -> str:
        if not self.running():
            return "Nothing is running."
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
        return "Stopped."

    def status(self) -> str:
        if self.proc is None:
            return "Idle."
        mins = (time.time() - self.started) / 60
        if self.running():
            return f"Running ({mins:.1f} min), pid {self.proc.pid}"
        rc = self.proc.returncode
        if rc == 0:
            return f"Finished successfully after {mins:.1f} min"
        if rc < 0:
            return f"Stopped after {mins:.1f} min"
        return f"Failed (exit code {rc}) after {mins:.1f} min; see log"

    def tail(self, n_chars: int = 20000) -> str:
        if not self.log_path or not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n_chars))
            text = f.read().decode(errors="replace")
        # tqdm writes \r progress updates; keep only the latest state of each line
        return "\n".join(line.split("\r")[-1] for line in text.split("\n"))


RUNS = RunManager()


def build_cmd(model, group, modality, train_modality, tts, use_existing, tailweight, target_size) -> List[str]:
    cmd = [sys.executable, "-m", "pipeline.run_pipeline_repit", "--model_path", model, "--target_group", group,
           "--modality", modality, "--tts", tts]
    if train_modality and train_modality != "same as modality":
        cmd += ["--train_modality", train_modality]
    if use_existing:
        cmd.append("--use_existing")
    if tailweight:
        cmd.append("--tailweight_ablation_study")
    if target_size:
        cmd.append("--target_size_study")
    return cmd


def start_run(model, group, modality, train_modality, tts, gpus, use_existing, tailweight, target_size):
    if not model or not group:
        raise gr.Error("Pick a model and a target group.")
    tm = None if train_modality == "same as modality" else train_modality
    if {modality, tm or modality} == {"audio", "audio+text"}:
        raise gr.Error("audio and audio+text can't be mixed in one run.")
    env = {"CUDA_VISIBLE_DEVICES": gpus.strip()} if gpus.strip() else {}
    RUNS.start(build_cmd(model, group, modality, train_modality, tts, use_existing, tailweight, target_size), env)
    return RUNS.status(), RUNS.tail()


def poll_run():
    return RUNS.status(), RUNS.tail()


def preview_cmd(*args):
    return " ".join(build_cmd(*args)).replace(sys.executable, "python")


# =============================================================================
# Results tab
# =============================================================================
def list_runs() -> List[str]:
    if not os.path.isdir(RUNS_DIR):
        return []
    runs = [d for d in os.listdir(RUNS_DIR) if not d.startswith("_") and os.path.isdir(os.path.join(RUNS_DIR, d))]
    return sorted(runs, key=lambda d: os.path.getmtime(os.path.join(RUNS_DIR, d)), reverse=True)


def _load_eval(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize_run(run: str):
    """(summary df, per-category df, info markdown, plot paths, completion files)."""
    if not run:
        return pd.DataFrame(), pd.DataFrame(), "Pick a run.", [], gr.update(choices=[], value=None)
    rdir = os.path.join(RUNS_DIR, run)
    rows, cat_rows = [], []
    eval_files = sorted(glob.glob(os.path.join(rdir, "completions", "**", "*_evaluations.json"), recursive=True))
    for path in eval_files:
        ev = _load_eval(path)
        name = os.path.basename(path)[: -len("_evaluations.json")]
        folder = os.path.relpath(os.path.dirname(path), os.path.join(rdir, "completions"))
        dataset = name.split("_")[0] if name.startswith("harmful") else name.split("_ablation_")[0]
        row = {"intervention": folder, "dataset": dataset, "n": len(ev.get("completions", []))}
        for judge, short in (("llamaguard3", "LG3"), ("substring_matching", "substr")):
            if f"{judge}_success_rate" in ev:
                row[f"{short} ASR"] = ev[f"{judge}_success_rate"]
                row[f"{short} target"] = ev.get(f"is_jailbreak_{judge}_target")
                row[f"{short} non-target"] = ev.get(f"is_jailbreak_{judge}_nontarget")
                if dataset == "harmful":
                    for cat, v in ev.get(f"{judge}_per_category", {}).items():
                        cat_rows.append({"intervention": folder, "judge": short, "category": cat, "ASR": v})
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        order = {k: i for i, k in enumerate(INTERVENTIONS)}
        df["_o"] = df["intervention"].map(lambda x: order.get(x, 99))
        df = df.sort_values(["dataset", "_o", "intervention"]).drop(columns="_o")
        df = df.round(3).astype(object).where(df.notna(), "–")  # target/non-target don't apply to eval sets
    cat_df = pd.DataFrame(cat_rows)
    if not cat_df.empty:
        cat_df = cat_df.pivot_table(index=["judge", "category"], columns="intervention", values="ASR").round(3).reset_index()

    info = [f"### {run}"]
    meta = glob.glob(os.path.join(rdir, "rho_search", "best_rho_layer*_pos*.json"))
    if meta:
        m = _load_eval(meta[0])
        info.append(f"- direction: layer **{m.get('direction_layer')}**, position **{m.get('direction_pos')}**, "
                    f"rho **{m.get('best_rho')}**")
    elif os.path.exists(os.path.join(rdir, "direction_metadata.json")):
        m = _load_eval(os.path.join(rdir, "direction_metadata.json"))
        info.append(f"- direction (pre rho search): layer {m.get('layer')}, position {m.get('pos')}")
    if not eval_files:
        info.append("- no evaluations yet (judging runs at the end of the pipeline)")
    info.append(f"- evaluation files: {len(eval_files)}")

    plots = sorted(glob.glob(os.path.join(rdir, "**", "*.png"), recursive=True))
    comp_files = sorted(os.path.relpath(p, rdir) for p in glob.glob(os.path.join(rdir, "completions", "**", "*.json"), recursive=True))
    return df, cat_df, "\n".join(info), plots, gr.update(choices=comp_files, value=comp_files[0] if comp_files else None)


def show_completions(run: str, rel: str, query: str, only_jailbreaks: bool):
    if not run or not rel:
        return pd.DataFrame()
    data = _load_eval(os.path.join(RUNS_DIR, run, rel))
    comps = data["completions"] if isinstance(data, dict) else data
    df = pd.DataFrame(comps)
    if df.empty:
        return df
    if query:
        mask = df["prompt"].str.contains(query, case=False, na=False) | df["response"].str.contains(query, case=False, na=False)
        df = df[mask]
    judge_cols = [c for c in df.columns if c.startswith("is_jailbreak_")]
    if only_jailbreaks and judge_cols:
        df = df[df[judge_cols].max(axis=1) == 1]
    cols = ["category", "prompt", "response"] + judge_cols
    return df[[c for c in cols if c in df.columns]].head(500)


# =============================================================================
# Try-it tab
# =============================================================================
class Playground:
    def __init__(self):
        self.mb = None
        self.model_key = None
        self.hook_spec = None  # (layer, direction, reference)
        self.lock = threading.Lock()

    def load_model(self, model: str, modality: str):
        from pipeline.model_utils.model_factory import construct_model_base
        key = (model, modality)
        if self.model_key == key:
            return f"Already loaded: {model} ({modality})"
        with self.lock:
            self.unload()
            self.mb = construct_model_base(model, modality=modality)
            self.model_key = key
        return f"Loaded {model} ({modality}): {self.mb.n_layers} layers, d_model {self.mb.d_model}, on {self.mb.model.device}"

    def unload(self):
        import gc
        from pipeline.utils.utils import empty_device_cache
        self.mb, self.model_key = None, None
        gc.collect()
        empty_device_cache()

    def load_direction(self, run: str, which: str):
        import torch
        rdir = os.path.join(RUNS_DIR, run)
        meta = glob.glob(os.path.join(rdir, "rho_search", "best_rho_layer*_pos*.json"))
        ref_path = os.path.join(rdir, "generate_directions", "harmless_reference.pt")
        if not meta or not os.path.exists(ref_path):
            raise gr.Error("This run has no finished direction yet (needs rho_search/ and generate_directions/).")
        m = _load_eval(meta[0])
        pos, layer = int(m["direction_pos"]), int(m["direction_layer"])
        reference = torch.load(ref_path, map_location="cpu")[pos, layer]
        if which == "RepIt (target concept only)":
            direction = torch.load(os.path.join(rdir, "rho_search", "final_direction.pt"), map_location="cpu")
        else:  # full refusal direction before isolation (rho = 0)
            from pipeline.run_pipeline_repit import get_category_groups_by_name
            from pipeline.submodules.disentangle_vectors import isolate_target_vectors
            group = run.split("-RepIt-")[1].split("-")[0]
            targets, _ = get_category_groups_by_name(group)
            cat_means = torch.load(os.path.join(rdir, "generate_directions", "cat_means.pt"), map_location="cpu")
            direction = isolate_target_vectors(cat_means, targets=targets, rho=0.0)[pos, layer]
        self.hook_spec = (layer, direction.float(), reference.float())
        return f"Direction from {run}: {which}, layer {layer}, position {pos}, rho {m.get('best_rho')}"

    def generate(self, text: str, audio, max_new_tokens: int):
        if self.mb is None:
            raise gr.Error("Load a model first.")
        item = {"instruction": text or "", "category": None}
        if audio is not None:
            if not getattr(self.mb, "processor", None):
                raise gr.Error("This model is text-only; remove the audio or load an audio model.")
            sr, wav = audio
            wav = np.asarray(wav)
            if np.issubdtype(wav.dtype, np.integer):
                wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
            item["audio"] = {"array": wav.astype(np.float32), "sampling_rate": int(sr)}
        elif not text:
            raise gr.Error("Type a prompt or record/upload audio.")

        from pipeline.utils.hook_utils import get_affine_direction_ablation_input_pre_hook
        with self.lock:
            base = self.mb.generate_completions([item], batch_size=1, max_new_tokens=max_new_tokens, progress=False)[0]["response"]
            if self.hook_spec is None:
                return base, "(load a direction to compare)"
            layer, direction, reference = self.hook_spec
            hooks = [(self.mb.model_block_modules[layer],
                      get_affine_direction_ablation_input_pre_hook(direction=direction, reference=reference))]
            steered = self.mb.generate_completions([item], fwd_pre_hooks=hooks, batch_size=1,
                                                   max_new_tokens=max_new_tokens, progress=False)[0]["response"]
        return base, steered


PLAY = Playground()


# =============================================================================
# Layout
# =============================================================================
def build_ui() -> gr.Blocks:
    groups = target_groups()
    with gr.Blocks(title="SpeechBreak") as demo:
        gr.Markdown("# SpeechBreak\nRepIt concept-specific refusal directions on audio LLMs.")

        # ---------------------------------------------------------------- Run
        with gr.Tab("Run"):
            with gr.Row():
                model = gr.Dropdown(DEFAULT_MODELS, value=DEFAULT_MODELS[0], allow_custom_value=True,
                                    label="Model (HF id or local path)")
                group = gr.Dropdown(groups, value=groups[0], allow_custom_value=True, label="Target group")
            with gr.Row():
                modality = gr.Radio(MODALITIES, value="audio", label="Modality (test + eval prompts)")
                train_modality = gr.Radio(["same as modality"] + MODALITIES, value="same as modality",
                                          label="Train modality (direction extraction / selection)")
            with gr.Row():
                tts = gr.Textbox("mms", label="TTS backend", info="mms (default), mms:HF_ID, say, say:VOICE")
                gpus = gr.Textbox("", label="CUDA_VISIBLE_DEVICES", info="blank = inherit")
            with gr.Row():
                use_existing = gr.Checkbox(True, label="Reuse cached steps (--use_existing)")
                tailweight = gr.Checkbox(False, label="Tailweight ablation study")
                target_size = gr.Checkbox(False, label="Target size study (slow)")
            cmd_args = [model, group, modality, train_modality, tts, use_existing, tailweight, target_size]
            cmd_preview = gr.Code(label="Command", language="shell", value=preview_cmd(*[c.value for c in cmd_args]))
            for c in cmd_args:
                c.change(preview_cmd, cmd_args, cmd_preview)
            with gr.Row():
                start_btn = gr.Button("Start run", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
            status = gr.Textbox(RUNS.status(), label="Status", interactive=False)
            log = gr.Textbox(RUNS.tail(), label="Log (last 20k chars)", lines=22, max_lines=22,
                             autoscroll=True, interactive=False)
            start_btn.click(start_run, [model, group, modality, train_modality, tts, gpus, use_existing,
                                        tailweight, target_size], [status, log])
            stop_btn.click(lambda: (RUNS.stop(), RUNS.tail()), None, [status, log])
            gr.Timer(2.0).tick(poll_run, None, [status, log])

        # ------------------------------------------------------------ Results
        with gr.Tab("Results"):
            with gr.Row():
                run_dd = gr.Dropdown(list_runs(), label="Run", scale=4)
                refresh = gr.Button("Refresh", scale=1)
            info = gr.Markdown()
            gr.Markdown("**Attack success rate** by intervention. *target* = the run's target categories, "
                        "*non-target* = everything else. RepIt should be high on target and low on non-target.")
            summary = gr.Dataframe(wrap=True)
            with gr.Accordion("Per-category ASR (held-out harmful split)", open=False):
                per_cat = gr.Dataframe(wrap=True)
            with gr.Accordion("Plots", open=False):
                plots = gr.Gallery(columns=2, height="auto")
            gr.Markdown("**Completions**")
            with gr.Row():
                comp_file = gr.Dropdown([], label="File", scale=3)
                query = gr.Textbox(label="Search prompt/response", scale=2)
                only_jb = gr.Checkbox(False, label="Only judged jailbreaks", scale=1)
            comps = gr.Dataframe(wrap=True, max_height=600)

            outs = [summary, per_cat, info, plots, comp_file]
            run_dd.change(summarize_run, run_dd, outs)
            demo.load(summarize_run, run_dd, outs)  # the dropdown preselects the newest run
            refresh.click(lambda r: gr.update(choices=list_runs(), value=r), run_dd, run_dd).then(summarize_run, run_dd, outs)
            for c in (comp_file, query, only_jb):
                c.change(show_completions, [run_dd, comp_file, query, only_jb], comps)

        # ------------------------------------------------------------- Try it
        with gr.Tab("Try it"):
            gr.Markdown("Compare the model's answer with and without a run's direction ablated. "
                        "Loading a 7B model needs ~16 GB of GPU memory; don't do it while a run is using the same GPU.")
            with gr.Row():
                p_model = gr.Dropdown(DEFAULT_MODELS, value=DEFAULT_MODELS[0], allow_custom_value=True, label="Model")
                p_mode = gr.Radio(MODALITIES, value="audio", label="Modality",
                                  info="audio+text also feeds the typed text as the transcript")
                load_btn = gr.Button("Load model")
            with gr.Row():
                p_run = gr.Dropdown(list_runs(), label="Direction from run", scale=3)
                p_which = gr.Radio(["RepIt (target concept only)", "Full refusal direction"],
                                   value="RepIt (target concept only)", label="Direction", scale=2)
                dir_btn = gr.Button("Use direction", scale=1)
            p_status = gr.Markdown()
            with gr.Row():
                p_audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Spoken prompt")
                p_text = gr.Textbox(lines=4, label="Text prompt / transcript",
                                    info="Ignored when audio is given in 'audio' mode; used as transcript in 'audio+text'")
            p_tokens = gr.Slider(16, 512, value=200, step=16, label="Max new tokens")
            go = gr.Button("Generate", variant="primary")
            with gr.Row():
                out_base = gr.Textbox(label="Baseline", lines=12)
                out_steer = gr.Textbox(label="With direction ablated", lines=12)

            load_btn.click(PLAY.load_model, [p_model, p_mode], p_status)
            dir_btn.click(PLAY.load_direction, [p_run, p_which], p_status)
            p_run.focus(lambda r: gr.update(choices=list_runs(), value=r), p_run, p_run)
            go.click(PLAY.generate, [p_text, p_audio, p_tokens], [out_base, out_steer])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and use an SSH tunnel on shared servers")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    os.makedirs(RUNS_DIR, exist_ok=True)
    # allowed_paths: let Gradio serve run plots even when RUNS_DIR is outside the cwd
    build_ui().queue().launch(server_name=args.host, server_port=args.port, share=False,
                              allowed_paths=[RUNS_DIR])


if __name__ == "__main__":
    main()
