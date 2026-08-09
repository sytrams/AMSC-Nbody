# Cluster GoogleTest container

This Apptainer/Singularity image provides the Linux toolchain required by the
AMSC N-body CMake test targets:

- CUDA 12.8 development toolkit and Compute Sanitizer;
- Clang 18 for the C++23 `particle` module;
- CMake, Ninja, and the system GoogleTest package.

The source tree is not copied into the image. Apptainer bind-mounts the project
directory at runtime, so the same image can test any commit without rebuilding
the container.

## Build the image

From the project root, on a machine where unprivileged/fakeroot builds are
enabled:

```bash
chmod +x containers/build-container.sh containers/run-google-tests.sh
containers/build-container.sh nbody-tests.sif --fakeroot
```

If the cluster disables local image builds, run that command on a Linux machine
with Apptainer and copy `nbody-tests.sif` to the cluster. A SIF is a single,
read-only file.

## Interactive runs

Run all CPU and CUDA GoogleTests from a GPU allocation:

```bash
apptainer run --nv nbody-tests.sif all "$PWD"
```

Run only the CUDA or CPU suite:

```bash
apptainer run --nv nbody-tests.sif cuda "$PWD"
apptainer run nbody-tests.sif cpu "$PWD"
```

The default build directories are `build/cluster-all`,
`build/cluster-cuda`, and `build/cluster-cpu`. Override the location with
`NBODY_BUILD_DIR`; this is useful when the source tree is read-only:

```bash
NBODY_BUILD_DIR="$SCRATCH/nbody-build" \
  apptainer run --nv nbody-tests.sif all "$PWD"
```

By default CUDA code is compiled for the GPU visible on the allocated node.
For a portable binary, set a semicolon-separated CMake architecture list, for
example:

```bash
NBODY_CUDA_ARCHITECTURES='80;90' \
  apptainer run --nv nbody-tests.sif all "$PWD"
```

The host NVIDIA driver must support CUDA 12.8 because Apptainer passes the host
driver into the container. If the cluster driver is older, change the `From:`
tag in `nbody-tests.def` to a compatible CUDA development image before building.

## Slurm

Edit the account/partition/resource directives in
`containers/slurm-google-tests.sbatch`, then submit from the project root:

```bash
sbatch containers/slurm-google-tests.sbatch
```

Useful overrides are:

```bash
NBODY_IMAGE=/path/to/nbody-tests.sif \
NBODY_TEST_MODE=cuda \
NBODY_SOURCE_DIR=/path/to/AMSC-Nbody \
  sbatch containers/slurm-google-tests.sbatch
```

Do not request a GPU for CPU-only jobs. In that case, run the image without
`--nv`, or copy the batch script and remove `--gres=gpu:1` and `--nv`.
