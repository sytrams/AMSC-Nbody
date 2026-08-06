#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVCC="${NVCC:-nvcc}"

BUILD_DIRECTORY="$ROOT/build_morton_groups"
BINARY="$BUILD_DIRECTORY/morton_leaf_groups_test"

mkdir -p "$BUILD_DIRECTORY"

"$NVCC" \
    -std=c++17 \
    -O0 \
    -G \
    -Xcompiler=-Wall \
    -Xcompiler=-Wextra \
    -I"$ROOT/src" \
    "$ROOT/gpu/cuda_kernels/morton_leaf_groups.cu" \
    "$ROOT/test_morton_groups/morton_leaf_groups_test.cu" \
    -o "$BINARY"

"$BINARY"

if command -v compute-sanitizer >/dev/null 2>&1; then
    compute-sanitizer \
        --tool memcheck \
        --leak-check full \
        "$BINARY"

    if [[ "${RUN_RACECHECK:-0}" == "1" ]]; then
        compute-sanitizer \
            --tool racecheck \
            "$BINARY"
    fi
else
    echo \
        "compute-sanitizer not found: memory checking was skipped." \
        >&2
fi