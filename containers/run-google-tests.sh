#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: run-google-tests [cpu|cuda|all|sanitizer] [source-directory]

Build and run the AMSC N-body GoogleTest suite.

Environment variables:
  NBODY_SOURCE_DIR          Project directory (default: current directory)
  NBODY_BUILD_DIR           Build directory (default: <source>/build/cluster-<mode>)
  NBODY_BUILD_TYPE          CMake build type (default: RelWithDebInfo)
  NBODY_CUDA_ARCHITECTURES  CMake CUDA architectures (default: native)
  NBODY_JOBS                Parallel build jobs (default: SLURM_CPUS_PER_TASK or nproc)
  NBODY_CMAKE_ARGS          Additional whitespace-separated CMake arguments
  NBODY_CTEST_ARGS          Additional whitespace-separated CTest arguments
EOF
}

mode="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${mode}" in
    -h|--help)
        usage
        exit 0
        ;;
    cpu|cuda|all|sanitizer)
        ;;
    *)
        printf 'Unknown test mode: %s\n\n' "${mode}" >&2
        usage >&2
        exit 2
        ;;
esac

source_dir="${1:-${NBODY_SOURCE_DIR:-${PWD}}}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ $# -gt 0 ]]; then
    printf 'Unexpected argument: %s\n\n' "$1" >&2
    usage >&2
    exit 2
fi

if [[ ! -f "${source_dir}/CMakeLists.txt" || ! -f "${source_dir}/tests/CMakeLists.txt" ]]; then
    printf 'No N-body CMake project found at: %s\n' "${source_dir}" >&2
    printf 'Pass the project directory as the second argument or set NBODY_SOURCE_DIR.\n' >&2
    exit 2
fi

build_dir="${NBODY_BUILD_DIR:-${source_dir}/build/cluster-${mode}}"
build_type="${NBODY_BUILD_TYPE:-RelWithDebInfo}"

if [[ -n "${NBODY_JOBS:-}" ]]; then
    jobs="${NBODY_JOBS}"
elif [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    jobs="${SLURM_CPUS_PER_TASK}"
else
    jobs="$(nproc)"
fi

enable_cuda=OFF
enable_sanitizers=OFF
ctest_filter=()

case "${mode}" in
    cpu)
        ctest_filter=(-L cpu-unit)
        ;;
    cuda)
        enable_cuda=ON
        ctest_filter=(-L 'cuda-(unit|integration)')
        ;;
    all)
        enable_cuda=ON
        ;;
    sanitizer)
        enable_cuda=ON
        enable_sanitizers=ON
        ctest_filter=(-L cuda-sanitizer)
        ;;
esac

cmake_args=(
    -S "${source_dir}"
    -B "${build_dir}"
    -G Ninja
    -DBUILD_TESTING=ON
    -DNBODY_BUILD_VIEWER=OFF
    -DNBODY_ENABLE_CUDA="${enable_cuda}"
    -DNBODY_ENABLE_COMPUTE_SANITIZER_TESTS="${enable_sanitizers}"
    -DCMAKE_BUILD_TYPE="${build_type}"
    -DCMAKE_CXX_COMPILER="${CXX:-clang++-18}"
)

if [[ "${enable_cuda}" == ON ]]; then
    cmake_args+=(
        -DCMAKE_CUDA_ARCHITECTURES="${NBODY_CUDA_ARCHITECTURES:-native}"
    )
fi

if [[ -n "${NBODY_CMAKE_ARGS:-}" ]]; then
    # This is intentionally shell-style whitespace splitting for simple -D options.
    read -r -a extra_cmake_args <<< "${NBODY_CMAKE_ARGS}"
    cmake_args+=("${extra_cmake_args[@]}")
fi

printf 'Configuring %s tests in %s\n' "${mode}" "${build_dir}"
cmake "${cmake_args[@]}"

printf 'Building with %s parallel job(s)\n' "${jobs}"
cmake --build "${build_dir}" --parallel "${jobs}"

ctest_args=(
    --test-dir "${build_dir}"
    --output-on-failure
    --no-tests=error
    "${ctest_filter[@]}"
)

if [[ -n "${NBODY_CTEST_ARGS:-}" ]]; then
    # This is intentionally shell-style whitespace splitting for simple CTest options.
    read -r -a extra_ctest_args <<< "${NBODY_CTEST_ARGS}"
    ctest_args+=("${extra_ctest_args[@]}")
fi

printf 'Running %s GoogleTests\n' "${mode}"
ctest "${ctest_args[@]}"
