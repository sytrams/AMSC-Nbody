#!/usr/bin/env bash

set -euo pipefail

BINARY="${NBODY_BINARY:-build/simulation/nbody}"
GENERATOR="${NBODY_GENERATOR:-data/generate_galaxy.py}"

STEPS="${NBODY_BENCHMARK_STEPS:-10}"
REPEATS="${NBODY_BENCHMARK_REPEATS:-3}"

THETA="${NBODY_BENCHMARK_THETA:-0.5}"
TIME_STEP="${NBODY_BENCHMARK_DT:-0.001}"
SOFTENING="${NBODY_BENCHMARK_SOFTENING:-1e-6}"

DATA_DIR="${NBODY_BENCHMARK_DATA_DIR:-build/benchmark-data}"
RESULTS="${NBODY_BENCHMARK_RESULTS:-build/benchmark-results/scaling_results.csv}"

SIZES=(
    10000
    50000
    100000
    250000
)

if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "Benchmark executable not found: ${BINARY}" >&2
    echo "Build it first with:" >&2
    echo "  cmake --build build/simulation --target nbody --parallel" >&2
    exit 1
fi

if [[ ! -f "${GENERATOR}" ]]; then
    echo "Galaxy generator not found: ${GENERATOR}" >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"
mkdir -p "$(dirname "${RESULTS}")"

echo \
"particles,repeat,steps,theta,initialization_seconds,evolution_seconds,average_step_ms,particle_steps_per_second" \
> "${RESULTS}"

echo "N-body scaling benchmark"
echo "Binary:      ${BINARY}"
echo "Steps:       ${STEPS}"
echo "Repeats:     ${REPEATS}"
echo "Theta:       ${THETA}"
echo "Time step:   ${TIME_STEP}"
echo "Softening:   ${SOFTENING}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
fi

echo

for particles in "${SIZES[@]}"; do
    input="${DATA_DIR}/spiral_${particles}.bin"

    if [[ ! -f "${input}" ]]; then
        echo "Generating ${particles} particles..."

        python3 "${GENERATOR}" \
            --galaxy spiral \
            --seed 42 \
            -n "${particles}" \
            -o "${input}"
    fi

    for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
        echo "N=${particles}, repetition ${repeat}/${REPEATS}"

        output="$(
            "${BINARY}" \
                --input "${input}" \
                --time-step "${TIME_STEP}" \
                --steps "${STEPS}" \
                --theta "${THETA}" \
                --softening "${SOFTENING}"
        )"

        initialization="$(
            awk '/initialization wall time:/ {print $(NF-1)}' <<< "${output}"
        )"

        evolution="$(
            awk '/evolution wall time:/ {print $(NF-1)}' <<< "${output}"
        )"

        average_ms="$(
            awk '/average step time:/ {print $(NF-1)}' <<< "${output}"
        )"

        if [[ -z "${initialization}" ||
              -z "${evolution}" ||
              -z "${average_ms}" ]]; then
            echo "Could not parse benchmark output:" >&2
            echo "${output}" >&2
            exit 1
        fi

        throughput="$(
            awk \
                -v n="${particles}" \
                -v steps="${STEPS}" \
                -v seconds="${evolution}" \
                'BEGIN { printf "%.6f", (n * steps) / seconds }'
        )"

        echo \
"${particles},${repeat},${STEPS},${THETA},${initialization},${evolution},${average_ms},${throughput}" \
        >> "${RESULTS}"
    done
done

echo
echo "Benchmark complete."
echo "Results written to ${RESULTS}"
