#!/usr/bin/env bash
set -euo pipefail

script_directory="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"
project_root="$(
    cd "$script_directory/../.."
    pwd
)"
build_directory="${1:-$project_root/build/cuda-tests}"

cmake_arguments=(
    -S "$project_root"
    -B "$build_directory"
    -G Ninja
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
cmake --build "$build_directory" --parallel --target nbody_cuda_unit_tests nbody_cuda_integration_tests

compute-sanitizer --tool memcheck --leak-check full "$build_directory/tests/nbody_cuda_unit_tests"

compute-sanitizer --tool memcheck --leak-check full "$build_directory/tests/nbody_cuda_integration_tests"

if [[ "${RUN_RACECHECK:-0}" == "1" ]]; then
    compute-sanitizer --tool racecheck "$build_directory/tests/nbody_cuda_unit_tests"

    compute-sanitizer --tool racecheck "$build_directory/tests/nbody_cuda_integration_tests"
fi
