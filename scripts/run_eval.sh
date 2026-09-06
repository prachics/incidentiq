#!/usr/bin/env bash
# Run the full evaluation suite, resuming automatically if a previous run was
# interrupted.
#
#   ./scripts/run_eval.sh              full suite, local model, judge on
#   ./scripts/run_eval.sh --sample 2   a stratified subset
#
# Safe to run repeatedly. If a progress file exists for this exact
# configuration it resumes; otherwise it starts fresh. An interrupted run loses
# at most the scenario that was in flight, never the whole suite.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate

PROGRESS_DIR="evals/results/progress"
RESUME=""
if compgen -G "$PROGRESS_DIR/*.jsonl" > /dev/null 2>&1; then
    DONE=$(cat "$PROGRESS_DIR"/*.jsonl 2>/dev/null | grep -c . || echo 0)
    echo "Found an interrupted run with $DONE scenario(s) already done — resuming."
    RESUME="--resume"
fi

# caffeinate -i prevents the machine idling to sleep while this runs. It cannot
# override sleep triggered by closing the lid; that pauses the run rather than
# losing it, and it continues when the lid is opened.
exec caffeinate -i python -u -m evals.run --provider ollama --model qwen2.5:14b $RESUME "$@"
