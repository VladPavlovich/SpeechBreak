#!/usr/bin/env bash
# Live web demos: talk to the models in your browser, unsteered or steered.
#
#   bash scripts/live.sh            # both: Qwen2.5-Omni on :7860, Kimi-Audio on :7861
#   bash scripts/live.sh qwen       # only Qwen2.5-Omni (speak or type -> spoken reply), ~25 GB GPU
#   bash scripts/live.sh kimi       # only Kimi-Audio   (speak or type -> spoken reply), ~33 GB GPU
#   QWEN_PORT=7870 KIMI_PORT=7871 bash scripts/live.sh
#
# The pages are served on 127.0.0.1 of this machine. From your laptop, tunnel the ports and open
# http://localhost:7860 / http://localhost:7861 (browsers only allow the microphone on localhost):
#   ssh -L 7860:localhost:7860 -L 7861:localhost:7861 <you>@<this-server>
#
# Ctrl+C stops both.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 LOGURU_LEVEL=WARNING

WHICH=${1:-both}
QWEN_PORT=${QWEN_PORT:-7860}
KIMI_PORT=${KIMI_PORT:-7861}
KIMI_ENV=${KIMI_ENV:-kimi}
declare -A NEED_MIB=([qwen]=25000 [kimi]=33000)

case "$WHICH" in
    qwen|kimi) MODELS=("$WHICH") ;;
    both)      MODELS=(qwen kimi) ;;
    *) echo "usage: bash scripts/live.sh [qwen|kimi|both]"; exit 2 ;;
esac

need=0; for m in "${MODELS[@]}"; do need=$(( need + NEED_MIB[$m] )); done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt "$need" ]; then
    echo "!!! ${MODELS[*]} need ~$((need / 1024)) GiB of GPU memory, only $((free / 1024)) GiB free:"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/      /'
    echo "    Run one model at a time (bash scripts/live.sh qwen | kimi) or wait for the GPU to free up."
    exit 1
fi

pids=()
trap 'kill "${pids[@]}" 2>/dev/null; exit 0' INT TERM
for m in "${MODELS[@]}"; do
    if [ "$m" = qwen ]; then
        python scripts/live_qwen.py --port "$QWEN_PORT" > >(sed -u "s/^/[qwen] /") 2>&1 &
    else
        conda run -n "$KIMI_ENV" --no-capture-output python scripts/live_kimi.py --port "$KIMI_PORT" \
            > >(sed -u "s/^/[kimi] /") 2>&1 &
    fi
    pids+=($!)
done

echo "Starting ${MODELS[*]} (loading takes ~1 min). Then, from your laptop:"
tunnel=""; urls=""
for m in "${MODELS[@]}"; do
    port=$([ "$m" = qwen ] && echo "$QWEN_PORT" || echo "$KIMI_PORT")
    tunnel+=" -L $port:localhost:$port"; urls+="  $m: http://localhost:$port"
done
echo "  ssh$tunnel $USER@$(hostname -f)"
echo "  then open$urls"
wait
