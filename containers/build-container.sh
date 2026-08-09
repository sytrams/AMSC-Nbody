#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
image="${1:-${project_dir}/nbody-tests.sif}"

if [[ $# -gt 0 ]]; then
    shift
fi

if command -v apptainer >/dev/null 2>&1; then
    runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
    runtime=singularity
else
    printf 'Apptainer or Singularity is required to build the image.\n' >&2
    exit 127
fi

cd "${project_dir}"
exec "${runtime}" build "$@" "${image}" containers/nbody-tests.def
