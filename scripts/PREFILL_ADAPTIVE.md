# Prefill adaptive bucket graph

Implemented for the Hygon SGLang source tree. The existing Qwen2-compatible
Router + Shared Expert segment adapter is used by Qwen3.5. The independent
`prefill_graph_policy.py` and callable-based executor do not select a model by
name; another MoE implementation still needs a segment adapter and validation.

## Execution

`source adaptive` selects the smallest eligible bucket from
`512,1024,1536,2048` (local hidden-state rows, not global prompt length).
The extra rows must be <=64 AND <=3% of actual rows. Demand from different
eligible sizes accumulates per bucket/layer; the third occurrence captures
on demand on the serving thread. Exact `graph-multi` behavior is preserved.

The graph input is persistent. Replay copies real input into its prefix,
zeros the remaining rows, and returns views of the real output prefix. There
are no attention, KV writes, routed experts or collective operations inside
this segment. Weights must remain immutable and execution must be serial on
the same stream. Restart after weight/LoRA changes. Concurrent streams are
rejected by the signature guard. This is bounded demand-driven capture, not
kernel parameter updates or an automatically learned cost optimizer.

The first two distinct padded sizes per bucket/layer are strictly checked
against unpadded eager outputs using real weights. A mismatch returns the
saved eager result and permanently disables that bucket for that layer;
the summary checker fails the run. This finite warmup check is not a proof
for every shape, model or quantization backend. Set
`SGLANG_PREFILL_GRAPH_PADDING_VERIFY_CALLS=64` for an exhaustive check of the
allowed padded sizes, if needed. Default validation is bounded to avoid a
separate costly diagnostic benchmark. Capture failures remain fatal because
a failed vendor graph capture can leave the device runtime unusable.

## Resources

Defaults: four buckets/layer, sum(bucket rows)<=5120/layer, at least 4096 MiB
free device memory before admission, 2048 MiB process/device graph allocation
admission threshold across all layers. The last threshold is based on positive
`memory_allocated` deltas at capture. It is not a hard reservation: one capture
can exceed it; private allocator pools and other allocations also affect it.
No cross-graph memory aliasing, VMM mapping or eviction is introduced.

`PREFILL_GRAPH_CONFIG` prints effective worker settings. Schema-v2
`PREFILL_GRAPH_STATS` includes cumulative padding overhead, validation failures,
disabled buckets, allocated/reserved capture deltas, original rows histograms
and useful replay rows. Periodic snapshots are emitted at the last discovered
MoE layer after its execution so complete forwards are aligned. A normal Python
exit emits a final snapshot; SIGKILL does not. Periodic output may omit a final
partial interval. Counters include startup/warmup; they are not measure-only.

## One end-to-end comparison

The original container `wangxf35` was absent on 2026-10-26 during implementation.
Restore an environment with the original Hygon torch, kernels and dependencies
and mount the source at the same path before using these commands. CPU-only
tests and syntax checks were run; GPU capture/numerics/throughput were not run.

Use one fresh log per service instance. Start the same source-off Decode and
Router for both variants. In the Prefill terminal, start the first variant:

```bash
ROOT=/mnt/data/wangxf35/sglang-das
BENCH=/mnt/data/wangxf35/Qwen3.5_benchmark/qwen3.5_1p1d
MODE=off  # repeat with adaptive after stopping this Prefill instance
LOG="$BENCH/prefill-test/prefill-server-${MODE}-$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$BENCH/prefill-test"
printf '%s\n' "$LOG"
bash "$ROOT/scripts/run_prefill_variant.sh" source "$MODE" -- \
  bash "$BENCH/prefill-25-single.sh" >"$LOG" 2>&1
```

Keep the launch script's existing `--disable-cuda-graph` for this comparison:
it disables the native full/PCG runners, while this opt-in local segment is
controlled independently. Do not enable native PCG during the A/B. The adapter
returns eager inside PCG context to avoid nested capture/tracing.

After health is ready, run in a second terminal using the EXACT printed log:

```bash
ROOT=/mnt/data/wangxf35/sglang-das
bash "$ROOT/scripts/bench_prefill_adaptive.sh" off /ABS/PATH/TO/OFF.log 1
# Stop off Prefill, launch MODE=adaptive with a NEW log, refresh Router
# registration if required, and wait for health. Keep Decode configuration fixed.
bash "$ROOT/scripts/bench_prefill_adaptive.sh" adaptive /ABS/PATH/TO/ADAPTIVE.log 1
```

The helper uses warm seed=41, measure seed=42, mixed4K input, output=1,
concurrency=10/80, 100/800 measured requests, tokenize-prompt and measure flush
through the existing test.sh. Optional last argument=2 repeats measurement
without restarting the service. It does not stop/start services for you.

```bash
python3 "$ROOT/scripts/summarize_prefill_graph_stats.py" \
  --check-adaptive /ABS/PATH/TO/ADAPTIVE.log
grep -aE 'PREFILL_GRAPH_CONFIG|PREFILL_BUCKET_ADMIT|PREFILL_BUCKET_DISABLED|PREFILL_ISLAND_NUMERICS' \
  /ABS/PATH/TO/ADAPTIVE.log
```

Accept correctness only if all numerical checks pass and all expected ranks
capture/replay; verify `cached_shape_sets` contains all 48 layers on each of
the four ranks for the main bucket. The checker requires capture, replay and
at least one padding check in every logged rank, and rejects numerical failures.
Inspect useful token hit rate, padding_overhead_ratio, disabled buckets and
per-rank memory. A >=30% coverage target is a hypothesis, not a pass guarantee.
Compare fresh off/adaptive input throughput and mean/P90/P99 TTFT. A difference
around 1-2% should be treated as inconclusive with a single run. If replay
coverage rises but throughput does not, use a short GPU profile to assess copy,
GEMM and communication costs before enlarging the capture region.

## Native PCG and xLLM

PCG defines graph boundaries; buckets define reuse across input shapes. They
are complementary. In this source, `TcPiecewiseCudaGraphBackend` uses Dynamo/FX
and splits MoE A2A for DeepEP/Mooncake. Startup compiles/captures configured
sizes. `capture_prefill_graph` also checks standard GQA across layers, with a
special hybrid-handler exception for the breakable backend. Therefore an
available PCG switch is not evidence that Qwen3.5 hybrid attention plus the
installed Hygon FP8 operators can run it unchanged.

Useful xLLM ideas adopted here: online demand capture, persistent buffers,
bucketed execution, dynamic content at fixed addresses, eager boundaries for
dynamic communication. Not implemented: VMM/shared physical backing, full
model capture, task-node updates, online cost learning or eviction. Those need
backend/runtime integration and measured benefit, and are separate from the
current local segment change.

Offline checks:

```bash
python3 "$ROOT/scripts/test_prefill_adaptive_policy.py" -v
bash -n "$ROOT/scripts/run_prefill_variant.sh" "$ROOT/scripts/bench_prefill_adaptive.sh"
```
