#!/usr/bin/env bash
# Run after starting the selected Prefill, the unchanged Decode, and Router.
# Usage: bash bench_prefill_adaptive.sh off|adaptive ABS_SERVER_LOG [REPEATS]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:?off or adaptive required}
SERVER_LOG=${2:?absolute server log required}
REPEATS=${3:-1}
BENCH_DIR=${BENCH_DIR:-/mnt/data/wangxf35/Qwen3.5_benchmark/qwen3.5_1p1d}
[[ "$MODE" == off || "$MODE" == adaptive ]] || exit 2
[[ "$SERVER_LOG" == /* && -f "$SERVER_LOG" ]] || { echo 'Supply the current absolute server log' >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || exit 2
for port in 30022 30023 30005; do
  curl --noproxy '*' -fsS --max-time 10 "http://10.103.240.25:$port/health" >/dev/null
done
cd "$BENCH_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
for PHASE in warm measure; do
  if [[ "$PHASE" == warm ]]; then SEED=41; RUNS=1; else SEED=42; RUNS=$REPEATS; fi
  for ((RUN=1; RUN<=RUNS; RUN++)); do
    # Different warm/measure seeds; identical measure data for both variants.
    env PYTHONPATH="$ROOT/python" bash "$BENCH_DIR/test.sh" \
      --phase "$PHASE" \
      --start-concurrency 10 --multiplier 8 --end-concurrency 80 \
      --prompt-multiple 10 --scenarios 9 \
      --random-range-ratio 0 --seed "$SEED" \
      --log-prefix "prefill-test/${PHASE}_source-${MODE}_mixed4k_${RUN_ID}_r${RUN}"
  done
done
if [[ "$MODE" == adaptive ]]; then
  # Let the scheduler print its cumulative final snapshot before summarizing:
  # a clean service stop also emits an atexit snapshot when Python exits normally.
  python3 "$ROOT/scripts/summarize_prefill_graph_stats.py" --check-adaptive "$SERVER_LOG"
fi
