#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export MAX_JOBS="${MAX_JOBS:-1}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
# install missing cudnn headers & ninja build tools
apt-get update
apt-get install -y libcudnn9-headers-cuda-12 ninja-build

# Python dependencies are declared in pyproject.toml extras.
# Keep backend + megatron together so setup does not prune runtime deps (e.g. vllm).
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd "${repo_root}"
uv sync --extra backend --extra megatron --frozen --active

# Optional: verify flash-attn availability (used by flash attention backends).
if [ "${ART_VERIFY_FLASH_ATTN:-0}" = "1" ]; then
    echo "Verifying flash-attn import..."
    if ! uv run python - <<'PY'
try:
    import flash_attn  # noqa: F401
    print("flash_attn import OK")
except Exception as exc:
    raise SystemExit(f"flash_attn import failed: {exc}")
PY
    then
        echo "flash-attn not available; flash backend may be disabled."
    fi
fi

# transformer-engine-torch is source-only on PyPI.
# Rebuild it against the active torch so ABI symbols match at runtime.
if [ "${ART_REBUILD_TRANSFORMER_ENGINE_TORCH:-1}" = "1" ]; then
    python_bin="$(uv run python - <<'PY'
import sys
print(sys.executable)
PY
)"
    site_packages="$("${python_bin}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

    te_lib_paths=(
        "${site_packages}/torch/lib"
        "${site_packages}/nvidia/cudnn/lib"
        "${site_packages}/nvidia/cuda_runtime/lib"
        "${site_packages}/nvidia/cublas/lib"
        "${site_packages}/nvidia/cusparse/lib"
        "${site_packages}/nvidia/cusparselt/lib"
        "${site_packages}/nvidia/cufft/lib"
        "${site_packages}/nvidia/curand/lib"
        "${site_packages}/nvidia/nccl/lib"
        "${site_packages}/nvidia/nvjitlink/lib"
        "${site_packages}/nvidia/cuda_cupti/lib"
        "${site_packages}/nvidia/cufile/lib"
        "${site_packages}/nvidia/nvshmem/lib"
    )

    ld_parts=()
    for path in "${te_lib_paths[@]}"; do
        if [ -d "${path}" ]; then
            ld_parts+=("${path}")
        fi
    done
    te_runtime_ld="$(IFS=:; echo "${ld_parts[*]}")"

    echo "Rebuilding transformer-engine-torch from source for ABI compatibility..."
    env \
        LD_LIBRARY_PATH="${te_runtime_ld}:${LD_LIBRARY_PATH:-}" \
        NVTE_FRAMEWORK="${NVTE_FRAMEWORK}" \
        MAX_JOBS="${MAX_JOBS}" \
        NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB}" \
        "${python_bin}" -m pip install \
        --no-deps \
        --force-reinstall \
        --no-build-isolation \
        --no-binary transformer-engine-torch \
        --no-cache-dir \
        "transformer-engine-torch==2.11.0"

    echo "Verifying transformer_engine.pytorch import..."
    env LD_LIBRARY_PATH="${te_runtime_ld}:${LD_LIBRARY_PATH:-}" \
        "${python_bin}" - <<'PY'
import transformer_engine.pytorch  # noqa: F401
print("transformer_engine.pytorch import OK")
PY
fi
