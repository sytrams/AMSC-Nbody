#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_directory/../.." && pwd)"
build_directory="${1:-$project_root/build/cuda-tests}"

# WSL exposes the Windows NVIDIA driver libraries here. Prefer them when the
# directory exists so Compute Sanitizer uses the same driver bridge as CUDA.
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cmake_arguments=(
    -S "$project_root"
    -B "$build_directory"
    -DBUILD_TESTING=ON
    -DNBODY_ENABLE_CUDA=ON
    -DNBODY_BUILD_VIEWER=OFF
)

if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    cmake_arguments+=(
        "-DCMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES"
    )
fi

cmake "${cmake_arguments[@]}"
cmake --build "$build_directory" --parallel

cuda_test_targets=(
    nbody_cuda_unit_tests
    nbody_cuda_integration_tests
)

for target in "${cuda_test_targets[@]}"; do
    executable="$build_directory/tests/$target"
    if [[ ! -x "$executable" ]]; then
        echo "CUDA test executable not found: $executable" >&2
        exit 1
    fi

    compute-sanitizer \
        --error-exitcode 1 \
        --tool memcheck \
        --leak-check full \
        "$executable"

    if [[ "${RUN_RACECHECK:-0}" == "1" ]]; then
        compute-sanitizer \
            --error-exitcode 1 \
            --tool racecheck \
            "$executable"
    fi
done
