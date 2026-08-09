#!/usr/bin/env bash
#SBATCH --job-name=amsc-nbody-tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=ci-results/slurm-%j.out

set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-$PWD}"
IMAGE_PATH="${IMAGE_PATH:-${SOURCE_DIR}/nbody-tests.sif}"
RESULTS_DIR="${RESULTS_DIR:-${SOURCE_DIR}/ci-results/${SLURM_JOB_ID:-manual}}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80}"
NBODY_ENABLE_COMPUTE_SANITIZER_TESTS="${NBODY_ENABLE_COMPUTE_SANITIZER_TESTS:-OFF}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-apptainer}"

if [[ -n "${CONTAINER_MODULE:-}" ]]; then
    if ! type module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
        source /etc/profile
    fi
    module load "${CONTAINER_MODULE}"
fi

if ! command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1; then
    if [[ "${CONTAINER_RUNTIME}" == "apptainer" ]] && command -v singularity >/dev/null 2>&1; then
        CONTAINER_RUNTIME="singularity"
    else
        echo "Container runtime not found: ${CONTAINER_RUNTIME}" >&2
        exit 1
    fi
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "Source directory not found: ${SOURCE_DIR}" >&2
    exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Container image not found: ${IMAGE_PATH}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

SCRATCH_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
BUILD_DIR="${SCRATCH_ROOT}/amsc-nbody-${SLURM_JOB_ID:-manual}"
mkdir -p "${BUILD_DIR}"

container_exec=(
    "${CONTAINER_RUNTIME}"
    exec
    --nv
    --cleanenv
    --bind "${SOURCE_DIR}:/workspace:ro"
    --bind "${BUILD_DIR}:/build"
    --bind "${RESULTS_DIR}:/results"
    "${IMAGE_PATH}"
)

"${container_exec[@]}" nvidia-smi

"${container_exec[@]}" cmake \
    -S /workspace \
    -B /build \
    -G Ninja \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DNBODY_ENABLE_COMPUTE_SANITIZER_TESTS="${NBODY_ENABLE_COMPUTE_SANITIZER_TESTS}" \
    2>&1 | tee "${RESULTS_DIR}/configure.log"

"${container_exec[@]}" cmake \
    --build /build \
    --parallel "${SLURM_CPUS_PER_TASK:-4}" \
    2>&1 | tee "${RESULTS_DIR}/build.log"

"${container_exec[@]}" ctest \
    --test-dir /build \
    --parallel 1 \
    --output-on-failure \
    --output-junit /results/ctest-results.xml \
    2>&1 | tee "${RESULTS_DIR}/ctest.log"
