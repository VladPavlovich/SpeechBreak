#!/usr/bin/env bash
# Steer Kimi-Audio-7B-Instruct with a harmful-vs-harmless direction — all six steps
# (see steering/run_steps_kimi.py). The Kimi version of scripts/run_steps.sh; results in results_kimi/.
#
#   1 extract     LLM + audio-head activations at the last prompt token   -> results_kimi/1_activations/
#   2 directions  harmful − harmless direction per layer + separability   -> results_kimi/2_directions/
#   3 sweep       ±direction at each layer, forward passes only; ranks layers -> results_kimi/3_layer_sweep/
#   4 generate    spoken replies at the top layers × ALPHAS × N_GEN prompts, for every experiment in EXPS
#                                                                         -> results_kimi/4_generations/
#   5 evaluate    Whisper + judge, best setting per experiment           -> results_kimi/5_evaluation/summary.md
#   6 demo        reply to the text in scripts/demo_prompt.txt            -> results_kimi/6_demo/demo.json
#
# Prompts are typed text (dataset/processed/*.json), so no TTS is needed.
#
#   nohup bash scripts/run_steps_kimi.sh > run_steps_kimi.out 2>&1 &
#   STEPS="demo" bash scripts/run_steps_kimi.sh                                  # just the demo
#   STEPS="generate evaluate" bash scripts/run_steps_kimi.sh                     # resume from step 4
#   RANK_BY=d_resist TOP_K=2 ALPHAS="0.5 1 2 4" N_GEN=10 bash scripts/run_steps_kimi.sh
#   EXPS="exp1_llm exp3_both" bash scripts/run_steps_kimi.sh                     # subset of experiments
#   DEMO_MODE=suppress DEMO_LAYER=18 DEMO_ALPHA=1 STEPS=demo bash scripts/run_steps_kimi.sh   # jailbreak demo
#
# Two conda envs: Kimi-Audio itself (steps 1, 3, 4, 6) runs in KIMI_ENV (default "kimi": torch 2.6,
# flash-attn, Moonshot's kimia_infer); the directions and the Whisper + judge scoring (steps 2, 5)
# run in the env this script is started from, the same one the Qwen pipeline uses.
#
# Kimi-Audio writes one reply at a time, so step 4 runs parallel workers under a private NVIDIA MPS
# daemon, in two passes that each fill the GPU: first LLM-only workers write the reply text and
# audio tokens (~20 GiB each: 3 on an A100 80GB), then detokenizer-only workers turn the tokens
# into wavs (~13 GiB each: 5). Override the counts with WORKERS=N / VOCODE_WORKERS=N, disable MPS
# with USE_MPS=0. Worker lines are prefixed [w0], [w1], ...

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 LOGURU_LEVEL=WARNING

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

KIMI_ENV=${KIMI_ENV:-kimi}
kimi_py() { conda run -n "$KIMI_ENV" --no-capture-output python -m steering.run_steps_kimi "$@"; }
main_py() { python -m steering.run_steps_kimi "$@"; }

# ---------------------------------------------------------------------------- GPU workers
WORKER_MIB=${WORKER_MIB:-22000}          # LLM only: 18.6 GiB of weights + context + margin
VOCODE_MIB=${VOCODE_MIB:-15500}          # detokenizer only: measured ~13 GiB per worker + margin
MAX_WORKERS=${MAX_WORKERS:-8}

n_workers() {  # n_workers <MiB per worker> <override>
    if [ -n "${2:-}" ]; then echo "$2"; return; fi
    local free n
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    n=$(( (free - 1024) / $1 ))
    [ "$n" -lt 1 ] && n=1
    [ "$n" -gt "$MAX_WORKERS" ] && n=$MAX_WORKERS
    echo "$n"
}

MPS_STARTED=0
start_mps() {
    [ "${USE_MPS:-1}" = 1 ] && command -v nvidia-cuda-mps-control >/dev/null || return 0
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/speechbreak-mps-$USER/pipe"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/speechbreak-mps-$USER/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if nvidia-cuda-mps-control -d 2>/dev/null; then
        MPS_STARTED=1; log "    MPS daemon started"
    else
        log "    MPS daemon not started (already running?) - continuing"
    fi
}
stop_mps() {
    [ "$MPS_STARTED" = 1 ] && echo quit | nvidia-cuda-mps-control
    MPS_STARTED=0
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
}

kill_tree() {  # kill all descendants of a pid (workers sit under process-substitution subshells)
    local child
    for child in $(pgrep -P "$1"); do
        kill_tree "$child"
        kill "$child" 2>/dev/null
    done
}

# Stopping this script (Ctrl+C / kill) also stops its workers and the MPS daemon
trap 'kill_tree $$; sleep 3; stop_mps; exit 130' INT TERM

run_workers() {  # run_workers <n> <args...>: n parallel shards (MPS), lines prefixed [wN]
    local n=$1; shift
    log "    $n parallel worker(s): $2"
    [ "$n" -gt 1 ] && start_mps
    local pids=() i rc=0
    for ((i = 0; i < n; i++)); do
        kimi_py "$@" --shard "$i" --num_shards "$n" > >(sed -u "s/^/    [w$i] /") 2>&1 &
        pids+=($!)
        sleep 30  # stagger model loading so memory peaks don't coincide
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { log "!!! worker $i failed"; rc=1; }
    done
    stop_mps
    return $rc
}

# ---------------------------------------------------------------------------- options
STEPS=${STEPS:-"extract directions sweep generate evaluate demo"}
N_SWEEP=${N_SWEEP:-8}
TOP_K=${TOP_K:-3}
RANK_BY=${RANK_BY:-kl}
ALPHAS=${ALPHAS:-"0.5 1 2"}
N_GEN=${N_GEN:-5}
EXPS=${EXPS:-"exp1_llm exp2_audio_head exp3_both exp4a_llm_to_audio_head exp4b_audio_head_to_llm"}
DEMO_EXP=${DEMO_EXP:-exp1_llm}
DEMO_MODE=${DEMO_MODE:-induce}

DEMO_ARGS=(--demo_exp "$DEMO_EXP" --demo_mode "$DEMO_MODE")
[ -n "${DEMO_LAYER:-}" ] && DEMO_ARGS+=(--demo_layer "$DEMO_LAYER")
[ -n "${DEMO_ALPHA:-}" ] && DEMO_ARGS+=(--demo_alpha "$DEMO_ALPHA")

# ---------------------------------------------------------------------------- steps
T_START=$(date +%s)
log "=== Kimi-Audio steering | steps: $STEPS ==="
status=0
for step in $STEPS; do
    log "--- $step ---"
    case "$step" in
        extract)    kimi_py --step extract 2>&1 | sed -u "s/^/    /" ;;
        directions) main_py --step directions 2>&1 | sed -u "s/^/    /" ;;
        sweep)      kimi_py --step sweep --n_sweep "$N_SWEEP" --top_k "$TOP_K" --rank_by "$RANK_BY" 2>&1 | sed -u "s/^/    /" ;;
        generate)   run_workers "$(n_workers "$WORKER_MIB" "${WORKERS:-}")" --step generate --n_sweep "$N_SWEEP" \
                        --alphas $ALPHAS --n_gen "$N_GEN" --experiments $EXPS \
                        && run_workers "$(n_workers "$VOCODE_MIB" "${VOCODE_WORKERS:-}")" --step vocode
                    [ $? -eq 0 ] || { log "!!! generate failed"; status=1; break; }
                    continue ;;
        evaluate)   main_py --step evaluate 2>&1 | sed -u "s/^/    /" ;;
        demo)       kimi_py --step demo "${DEMO_ARGS[@]}" 2>&1 | sed -u "s/^/    /" \
                        && main_py --step demo_asr 2>&1 | sed -u "s/^/    /" ;;
        *)          log "unknown step $step"; status=1; break ;;
    esac
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "!!! $step failed"; status=1; break; }
done

ELAPSED=$(( $(date +%s) - T_START ))
log "=== done in $(( ELAPSED / 3600 ))h$(( ELAPSED % 3600 / 60 ))m, exit status $status ==="
log "results: results_kimi/3_layer_sweep/layer_table_*.md, results_kimi/5_evaluation/summary.md, results_kimi/6_demo/demo.json"
exit $status
