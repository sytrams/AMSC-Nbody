#!/usr/bin/env bash

# Adjust account, partition, memory, and wall time for the target cluster.
#SBATCH --job-name=nbody-gtest
#SBATCH --output=nbody-gtest-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -Eeuo pipefail

project_dir="${NBODY_SOURCE_DIR:-${SLURM_SUBMIT_DIR}}"
image="${NBODY_IMAGE:-${project_dir}/nbody-tests.sif}"
mode="${NBODY_TEST_MODE:-all}"
results_dir="${NBODY_RESULTS_DIR:-${project_dir}/ci-results/${SLURM_JOB_ID:-manual}}"
build_dir="${NBODY_BUILD_DIR:-${TMPDIR:-/tmp}/nbody-build-${SLURM_JOB_ID:-manual}}"

if [[ -n "${NBODY_CONTAINER_MODULE:-}" ]]; then
    if ! type module >/dev/null 2>&1 && [[ -r /etc/profile ]]; then
        set +u
        source /etc/profile
        set -u
    fi
    module load "${NBODY_CONTAINER_MODULE}"
fi

runtime="${NBODY_CONTAINER_RUNTIME:-}"
if [[ -z "${runtime}" ]]; then
    if command -v apptainer >/dev/null 2>&1; then
        runtime=apptainer
    elif command -v singularity >/dev/null 2>&1; then
        runtime=singularity
    fi
fi

if ! command -v "${runtime}" >/dev/null 2>&1; then
    printf 'Container runtime is not available: %s\n' "${runtime:-not configured}" >&2
    exit 127
fi

test -r "${image}"
test -f "${project_dir}/CMakeLists.txt"
mkdir -p "${results_dir}" "${build_dir}"

record_exit_status() {
    local status=$?
    set +e
    printf '%s\n' "${status}" > "${results_dir}/slurm-exit-code"
}
trap record_exit_status EXIT

srun "${runtime}" run \
    --nv \
    --cleanenv \
    --bind "${project_dir}:/workspace:ro" \
    --bind "${build_dir}:/build" \
    --bind "${results_dir}:/results" \
    --env NBODY_JOBS="${SLURM_CPUS_PER_TASK}" \
    --env NBODY_BUILD_DIR=/build \
    --env NBODY_CUDA_ARCHITECTURES="${NBODY_CUDA_ARCHITECTURES:-native}" \
    --env "NBODY_CTEST_ARGS=--parallel 1 --output-junit /results/ctest-results.xml" \
    "${image}" \
    "${mode}" \
    /workspace \
    2>&1 | tee "${results_dir}/google-tests.log"
