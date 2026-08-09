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

if command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
    runtime=singularity
else
    printf 'Apptainer or Singularity is not available on this node.\n' >&2
    exit 127
fi

test -r "${image}"
test -f "${project_dir}/CMakeLists.txt"

srun "${runtime}" run \
    --nv \
    --cleanenv \
    --env NBODY_JOBS="${SLURM_CPUS_PER_TASK}" \
    "${image}" \
    "${mode}" \
    "${project_dir}"
