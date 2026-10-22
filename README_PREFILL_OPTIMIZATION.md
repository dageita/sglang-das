# Prefill optimization integration

Target: `/mnt/data/wangxf35/sglang-das`, base commit `a8543c8131`.
Source: `/mnt/data/wangxf35/prefill-optimization`, `coverage_v1` implementation.

The model and TopK files contain incremental edits, retaining the newer DAS
implementation. Three helpers and eight regression scripts were imported. All
optimization switches default to off. Fused TopK64 is experimental (exact IDs,
weight tolerance rtol=2e-5, atol=2e-6), not bitwise-equivalent routing weights.
No DeepEP, DeepGEMM, weights or installed Python packages are replaced.

## Run from source without installation

Run inside container `wangxf35`. This is the simplest Python-only development
workflow; Triton kernels compile on first use. Source/native extension compatibility
must still be checked because this checkout is newer than the installed wheel.

```bash
ROOT=/mnt/data/wangxf35/sglang-das
bash "$ROOT/scripts/run_prefill_variant.sh" source off -- python3 -c 'import sglang; print(sglang.__file__)'
```

The launcher accepts `installed|legacy|source`, followed by
`off|graph|coverage|fused`, then `--` and a command. `installed` accepts only `off`.
`graph` enables only the original shape32 island. `coverage` enables multi-shape
graphs, single TopK postprocess and direct pad/cast. `fused` additionally enables
fused TopK64. Audit and numerical-validation variables are preserved; set them to
zero for timing. The launcher clears inherited PYTHONPATH and changes cwd to /tmp;
use absolute paths. Do not nest the old start_prefill.sh: it overwrites PYTHONPATH.

Use the existing *base* Prefill script, which does not override PYTHONPATH:

```bash
BASE=/mnt/data/wangxf35/Qwen3.5_benchmark/qwen3.5_1p1d/prefill-25-1.sh
# Run separately, restarting only the experiment Prefill between arms.
bash "$ROOT/scripts/run_prefill_variant.sh" source off -- bash "$BASE"
bash "$ROOT/scripts/run_prefill_variant.sh" source coverage -- bash "$BASE"
# Historical package comparisons (also separate runs):
bash "$ROOT/scripts/run_prefill_variant.sh" installed off -- bash "$BASE"
bash "$ROOT/scripts/run_prefill_variant.sh" legacy coverage -- bash "$BASE"
```

`source off` vs `source coverage` isolates the optimization bundle on the same
source base. `source off` vs `source graph` isolates the original graph island.
Installed-vs-source differences also include the framework revision. For independent
single/direct TopK ablations set the environment explicitly after the launcher,
using `-- env SGLANG_...=... bash "$BASE"`.
Hold model, P/D topology, dependencies, cache policy and workload fixed; verify zero
cached tokens, warm every measured shape, and exclude capture/JIT from steady-state
timing. Keep numerical validation separate. Restarted Prefill must be healthy in
the router before requests. Do not launch overlapping services on the same ports.

## Editable installation in an isolated environment

The existing DTK torch/triton/DeepEP/DeepGEMM wheels should remain in place.
Use a venv with system-site-packages; never globally reinstall upstream torch.

```bash
ROOT=/mnt/data/wangxf35/sglang-das
python3 -m venv --system-site-packages /mnt/data/wangxf35/venvs/sglang-das-prefill
source /mnt/data/wangxf35/venvs/sglang-das-prefill/bin/activate
unset PYTHONPATH
cd "$ROOT/python"
# Python 3.10 must use the HCU metadata; restore upstream metadata even on failure.
PYPROJECT_BACKUP=$(mktemp "$ROOT/python/pyproject.toml.backup.XXXXXX")
cp pyproject.toml "$PYPROJECT_BACKUP"
trap 'cp "$PYPROJECT_BACKUP" pyproject.toml; rm -f "$PYPROJECT_BACKUP"' EXIT
cp pyproject_hcu.toml pyproject.toml
# No dependency resolution; preserve the container DTK software stack.
SGLANG_BUILD_RUST_EXTS=none python -m pip install -e . --no-deps --no-build-isolation
cp "$PYPROJECT_BACKUP" pyproject.toml; rm -f "$PYPROJECT_BACKUP"; trap - EXIT
python -c 'import sglang; print(sglang.__file__)'
```

Build prerequisites include setuptools, wheel and setuptools-scm (setuptools-rust
when building Rust extensions). Install missing build tools inside this venv only.
`--no-deps` preserves the DTK stack but does not prove all new source dependencies
are satisfied. Diagnose import failures before starting the server.
The launcher selects system Python/CLI via PATH for installed and legacy modes;
source mode keeps the active interpreter. Do not pass an explicit venv Python
executable for an installed baseline. For manual commands use `/usr/local/bin/python3`
with `env -u PYTHONPATH` from /tmp for the installed baseline.

## Native HCU kernel build, only when needed

This patch changes Python/Triton only. If the new checkout requires native symbols
missing from the existing sglang-kernel, build the matching AOT package in the venv.
The repository's native build entry is `python/sglang/kernels/aot/setup_hip.py`.

```bash
source /mnt/data/wangxf35/venvs/sglang-das-prefill/bin/activate
cd /mnt/data/wangxf35/sglang-das/python/sglang/kernels/aot
python setup_hip.py bdist_wheel
# Install the specific newly built wheel printed by the build, not a broad glob:
python -m pip install --no-deps /absolute/path/to/newly-built-sglang-kernel.whl
```

DTK/hipcc and Torch ABI must match this container. This does not rebuild DeepEP or
DeepGEMM. Keep the installed baseline interpreter outside the venv for comparisons.
For a Python wheel, perform the same temporary `pyproject_hcu.toml` swap shown
above, then run `SGLANG_BUILD_RUST_EXTS=none python -m build --wheel
--no-isolation -Cfeatures=all_hip` from the repository python directory. Restore
`pyproject.toml`, then install the exact wheel with `python -m pip install --no-deps
/path/to/sglang.whl` in a separate venv. Omit the Rust override only when its
toolchain and extensions are intentionally required.

## Regression commands

```bash
cd /mnt/data/wangxf35/sglang-das
export PYTHONPATH="$PWD/python:$PWD/python/sglang/srt/batch_overlap"
HIP_VISIBLE_DEVICES='' python3 -m unittest discover -s test/prefill_optimization -p test_cache_policy.py
HIP_VISIBLE_DEVICES='' python3 -m unittest discover -s test/prefill_optimization -p test_numerical_checks.py
HIP_VISIBLE_DEVICES='' python3 test/prefill_optimization/test_coverage_policy.py
# Only on a confirmed idle device; example uses physical device 1.
HIP_VISIBLE_DEVICES=1 python3 test/prefill_optimization/test_graph.py --device 0
HIP_VISIBLE_DEVICES=1 python3 test/prefill_optimization/test_topk_single.py
HIP_VISIBLE_DEVICES=1 SGLANG_HCU_DIRECT_TOPK_POSTPROCESS=1 python3 test/prefill_optimization/test_topk_single.py
HIP_VISIBLE_DEVICES=1 python3 test/prefill_optimization/test_topk_direct.py
HIP_VISIBLE_DEVICES=1 python3 test/prefill_optimization/test_direct_large.py
HIP_VISIBLE_DEVICES=1 python3 test/prefill_optimization/test_fused_topk64.py
```

These tests do not establish full-model accuracy or performance for this source
revision. Run the existing real-model numerical audit and fixed-workload A/B before
promoting the optimized configuration.
