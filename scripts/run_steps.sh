#!/usr/bin/env bash
# Steer Qwen2.5-Omni with a harmful-vs-harmless direction — all six steps (see steering/run_steps.py)
#
#   1 extract     Thinker + Talker activations at the last audio token  -> results/1_activations/
#   2 directions  harmful − harmless direction per layer + separability -> results/2_directions/
#   3 sweep       ±direction at each layer, forward passes only, no generation; ranks layers
#                                                                       -> results/3_layer_sweep/
#   4 generate    spoken replies at the top layers × ALPHAS × N_GEN prompts, for every experiment in EXPS
#                                                                       -> results/4_generations/
#   5 evaluate    Whisper + judge, best setting per experiment         -> results/5_evaluation/summary.md
#   6 demo        reply to the text in scripts/demo_prompt.txt          -> results/6_demo/demo.json
#
#   nohup bash scripts/run_steps.sh > run_steps.out 2>&1 &
#   STEPS="demo" bash scripts/run_steps.sh                                    # just the demo
#   STEPS="generate evaluate" bash scripts/run_steps.sh                       # resume from step 4
#   RANK_BY=d_resist TOP_K=2 ALPHAS="0.5 1 2 4" N_GEN=10 bash scripts/run_steps.sh
#   EXPS="exp1_thinker exp3_both" bash scripts/run_steps.sh                   # subset of experiments
#   DEMO_MODE=suppress DEMO_LAYER=18 DEMO_ALPHA=1 STEPS=demo bash scripts/run_steps.sh   # jailbreak demo
#
# Steps resume: extract skips finished files, generate skips finished replies, evaluate skips scored ones.
#
# Qwen2.5-Omni speaks one reply at a time (one process uses ~25 GiB and a fraction of an A100), so
# step 4 starts as many workers as fit in free GPU memory (WORKER_MIB each, at most MAX_WORKERS)
# under a private NVIDIA MPS daemon; without MPS they would time-slice and give no speedup.
# Override with WORKERS=N, disable MPS with USE_MPS=0. Worker lines are prefixed [w0], [w1], ...

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------- GPU workers
WORKER_MIB=${WORKER_MIB:-25600}   # Qwen2.5-Omni measured peak + context + margin
MAX_WORKERS=${MAX_WORKERS:-8}

n_workers() {
    if [ -n "${WORKERS:-}" ]; then echo "$WORKERS"; return; fi
    local free n
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    n=$(( (free - 1024) / WORKER_MIB ))
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

run_workers() {  # run_workers <args...>: parallel shards (MPS), lines prefixed [wN]
    local n; n=$(n_workers)
    log "    $n parallel worker(s)"
    [ "$n" -gt 1 ] && start_mps
    local pids=() i rc=0
    for ((i = 0; i < n; i++)); do
        python -m steering.run_steps "$@" --shard "$i" --num_shards "$n" \
            > >(sed -u "s/^/    [w$i] /") 2>&1 &
        pids+=($!)
        sleep 20  # stagger model loading so memory peaks don't coincide
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
EXPS=${EXPS:-"exp1_thinker exp2_talker exp3_both exp4a_thinker_to_talker exp4b_talker_to_thinker"}
DEMO_EXP=${DEMO_EXP:-exp1_thinker}
DEMO_MODE=${DEMO_MODE:-induce}

DEMO_ARGS=(--demo_exp "$DEMO_EXP" --demo_mode "$DEMO_MODE")
[ -n "${DEMO_LAYER:-}" ] && DEMO_ARGS+=(--demo_layer "$DEMO_LAYER")
[ -n "${DEMO_ALPHA:-}" ] && DEMO_ARGS+=(--demo_alpha "$DEMO_ALPHA")
[ -n "${DEMO_VOICE:-}" ] && DEMO_ARGS+=(--demo_voice "$DEMO_VOICE")

# ---------------------------------------------------------------------------- steps
T_START=$(date +%s)
log "=== Qwen2.5-Omni steering | steps: $STEPS ==="
status=0
for step in $STEPS; do
    case "$step" in
        sweep)    extra=(--top_k "$TOP_K" --rank_by "$RANK_BY") ;;
        generate) extra=(--alphas $ALPHAS --n_gen "$N_GEN" --experiments $EXPS) ;;
        demo)     extra=("${DEMO_ARGS[@]}") ;;
        *)        extra=() ;;
    esac
    log "--- $step ---"
    if [ "$step" = generate ]; then
        run_workers --step generate --n_sweep "$N_SWEEP" "${extra[@]}" || { log "!!! generate failed"; status=1; break; }
        continue
    fi
    python -m steering.run_steps --step "$step" --n_sweep "$N_SWEEP" "${extra[@]}" 2>&1 | sed -u "s/^/    /"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "!!! $step failed"; status=1; break; }
done

ELAPSED=$(( $(date +%s) - T_START ))
log "=== done in $(( ELAPSED / 3600 ))h$(( ELAPSED % 3600 / 60 ))m, exit status $status ==="
log "results: results/3_layer_sweep/layer_table_*.md, results/5_evaluation/summary.md, results/6_demo/demo.json"
exit $status
