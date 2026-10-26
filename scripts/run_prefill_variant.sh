#!/usr/bin/env bash
# Usage: bash run_prefill_variant.sh installed|legacy|source off|graph|graph-multi|topk|coverage|fused -- command args...
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERSION=${1:?version required}
MODE=${2:?mode required}
shift 2
if [[ ${1:-} == -- ]]; then shift; fi
(( $# > 0 )) || { echo 'Supply a command, e.g. python3 -m sglang.launch_server ...' >&2; exit 2; }
unset PYTHONPATH
case "$VERSION" in
  installed)
    [[ "$MODE" == off ]] || { echo 'Installed baseline supports off only' >&2; exit 2; }
    export PATH="/usr/local/bin:/usr/bin:$PATH"
    unset VIRTUAL_ENV
    ;;
  legacy)
    export PATH="/usr/local/bin:/usr/bin:$PATH"
    unset VIRTUAL_ENV
    export PYTHONPATH=/mnt/data/wangxf35/prefill-optimization/coverage_v1/sgl
    ;;
  source) export PYTHONPATH="$ROOT/python" ;;
  *) echo 'version: installed|legacy|source' >&2; exit 2 ;;
esac
export SGLANG_PREFILL_ROUTER_SHARED_GRAPH=0
export SGLANG_PREFILL_GRAPH_MULTI_SHAPE=0
export SGLANG_HCU_SINGLE_TOPK_POSTPROCESS=0
export SGLANG_HCU_DIRECT_TOPK_POSTPROCESS=0
export SGLANG_HCU_FUSED_TOPK64=0
export SGLANG_PREFILL_ROUTER_SHARED_GRAPH_TARGET_TOKENS=32
export SGLANG_PREFILL_ROUTER_SHARED_GRAPH_MAX_TOKENS=128
export SGLANG_HCU_DIRECT_TOPK_MAX_ROWS=128
# Performance variants must never inherit expensive per-call diagnostics from
# the parent shell. Override after `--` with `env NAME=value` for a diagnostic run.
export SGLANG_PREFILL_ISLAND_AUDIT=0
export SGLANG_PREFILL_ISLAND_VALIDATE_CALLS=0
export SGLANG_HCU_FUSED_TOPK64_VALIDATE=0
export SGLANG_PREFILL_GRAPH_STATS_INTERVAL=128

enable_multi_shape_graph() {
    export SGLANG_PREFILL_ROUTER_SHARED_GRAPH=1
    export SGLANG_PREFILL_GRAPH_MULTI_SHAPE=1
    export SGLANG_PREFILL_ROUTER_SHARED_GRAPH_MAX_TOKENS=2048
    export SGLANG_PREFILL_GRAPH_MAX_SHAPES=4
    export SGLANG_PREFILL_GRAPH_TOKEN_BUDGET=2048
    export SGLANG_PREFILL_GRAPH_MIN_SEEN=3
    export SGLANG_PREFILL_GRAPH_MIN_TOKENS=8
    export SGLANG_PREFILL_GRAPH_FREE_MB=2048
}

enable_topk() {
    export SGLANG_HCU_SINGLE_TOPK_POSTPROCESS=1
    export SGLANG_HCU_DIRECT_TOPK_POSTPROCESS=1
    export SGLANG_HCU_DIRECT_TOPK_MAX_ROWS=65536
    export SGLANG_HCU_FUSED_TOPK64=1
}
case "$MODE" in
  off) ;;
  graph) export SGLANG_PREFILL_ROUTER_SHARED_GRAPH=1 ;;
  graph-multi) enable_multi_shape_graph ;;
  topk) enable_topk ;;
  coverage|fused)
    enable_multi_shape_graph
    export SGLANG_HCU_SINGLE_TOPK_POSTPROCESS=1
    export SGLANG_HCU_DIRECT_TOPK_POSTPROCESS=1
    export SGLANG_HCU_DIRECT_TOPK_MAX_ROWS=65536
    if [[ "$MODE" == fused ]]; then export SGLANG_HCU_FUSED_TOPK64=1; fi
    ;;
  *) echo 'mode: off|graph|graph-multi|topk|coverage|fused' >&2; exit 2 ;;
esac
echo "PREFILL_VARIANT version=$VERSION mode=$MODE root=$ROOT stats_interval=$SGLANG_PREFILL_GRAPH_STATS_INTERVAL"
# Keep cwd from shadowing the selected package. Pass absolute script paths.
cd /tmp
python3 -c 'import sglang; print("SGLang:", sglang.__file__, flush=True)'
exec "$@"
