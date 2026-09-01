# AMSC N-body

This project contains:

- a CUDA N-body simulator using a shared leapfrog/Verlet timestep and
  Barnes-Hut forces;
- a macOS Metal viewer for one static particle file;
- a cluster-hosted browser viewer for live and completed snapshot directories;
- a POSIX TCP server/client pair for transferring sampled simulation frames;
- Python tools for creating synthetic and Solar-System particle files.

The simulator and viewer are separate applications. In particular, the viewer
does not run the simulation and does not currently play the streamed
`.nbsnap` frame format.

## Physical model

For particles `q` and `k`, the gravitational force is

$$
f_{qk}(t) = -\frac{G m_q m_k}{|s_q(t) - s_k(t)|^3}
             [s_q(t) - s_k(t)].
$$

The code uses `G = 6.67430e-11`, so physically meaningful input data should use
kilograms, metres, metres per second, and seconds consistently. The
`--softening` value has the same length unit as the particle positions.

## Requirements and working directory

Run every command in this README from the directory containing this `README.md`
and `CMakeLists.txt`.

Common requirements are:

- CMake 3.28 or newer;
- Ninja for commands that use `-G Ninja` (omit `-G Ninja` to use another CMake
  generator);
- Python 3 and NumPy for the data-generation scripts.

Create an optional Python virtual environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy
```

The CUDA simulator additionally requires a Linux/POSIX environment, an NVIDIA
CUDA toolkit with `nvcc`, and a compatible NVIDIA GPU and driver at runtime.
The simulator cannot run through CUDA on macOS.

The viewer requires macOS, a Metal-capable Mac, and the Xcode command-line
tools. It does not require CUDA.

## Create input data

No generated `.bin` datasets are committed. Create one before running either
application.

For a viewer-friendly 100,000-particle Gaussian cloud:

```bash
python3 data/generate_galaxy.py \
  --galaxy globular \
  --num-particles 100000 \
  --output data/globular_cloud.bin
```

For a small physical simulator smoke test:

```bash
python3 data/generate_galaxy.py \
  --galaxy two-body \
  --output data/two_body.bin
```

The two-body command prints its orbital period and a suggested timestep for
128 steps per orbit. See [Generate synthetic data](#generate-synthetic-data)
for all generator options.

## CUDA simulator

### Build

Configure a release build with CUDA and the simulation executable enabled:

```bash
cmake -S . -B build/simulation -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DNBODY_ENABLE_CUDA=ON \
  -DNBODY_BUILD_SIMULATION=ON \
  -DNBODY_BUILD_STREAMING=ON \
  -DNBODY_BUILD_VIEWER=OFF
cmake --build build/simulation --target nbody --parallel
```

`NBODY_BUILD_STREAMING=ON` is required when the simulator is built. Building
the `nbody` target also builds the server executable needed by `--cluster`.

### Run

This complete two-body example uses a timestep close to the generator's
default 128-step recommendation:

```bash
./build/simulation/nbody \
  --input data/two_body.bin \
  --time-step 0.04 \
  --steps 128 \
  --theta 0.5 \
  --softening 0
```

The required options are:

- `-i, --input FILE`: particle input file;
- `-t, --time-step DT`: finite positive timestep;
- `-n, --steps N`: positive number of timesteps.

Force options are:

- `--theta VALUE`: non-negative Barnes-Hut opening angle; default `0.5`;
- `--softening VALUE`: non-negative softening length; default `1e-6`.

Run the following for the authoritative command-line summary, including all
streaming options:

```bash
./build/simulation/nbody --help
```

Without `--stream-dir` or `--cluster`, the simulator advances the state only in
memory and prints a completion summary. It does not write an updated full
particle `.bin` file. To retain sampled positions, use the frame-streaming
workflow below.

## macOS Metal viewer

The viewer displays one static full particle `.bin` file. It does not advance
time, animate a directory, follow the network client, or open `.nbsnap` files.

### Build

```bash
cmake -S . -B build/viewer -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DNBODY_ENABLE_CUDA=OFF \
  -DNBODY_BUILD_SIMULATION=OFF \
  -DNBODY_BUILD_STREAMING=OFF \
  -DNBODY_BUILD_VIEWER=ON
cmake --build build/viewer --target nbody_viewer --parallel
```

### Run

Always pass an existing input explicitly:

```bash
./build/viewer/nbody_viewer --file data/globular_cloud.bin
```

The complete viewer-specific options are:

- `-f FILE`, `--file FILE`, or `--file=FILE`: full particle file to display;
- `--step N` or `--step=N`: non-negative step label shown in the title;
- `--total-steps N` or `--total-steps=N`: positive total-step label shown in
  the title.

`--step` and `--total-steps` are display labels only; they do not select or load
a simulation step. If both are present, `--step` cannot exceed
`--total-steps`. For example:

```bash
./build/viewer/nbody_viewer \
  --file data/globular_cloud.bin \
  --step 0 \
  --total-steps 128
```

Viewer controls are:

- arrow keys: pan horizontally or vertically;
- Control+Plus or Control+Equals: zoom in;
- Control+Minus: zoom out;
- vertical two-finger scrolling: pan vertically in the intentionally mirrored
  direction.

Particle colours are yellow for stars, blue for planets, light gray for moons,
brown for asteroids, and white for legacy or unknown types.

## Cluster-hosted browser viewer

`nbody_browser_viewer` is a lightweight HTTP server plus a WebGL 2 renderer. It
does not consume a cluster GPU: the CUDA devices remain available to the
simulation, while the local browser renders the downloaded display sample on
the user's GPU. The server can run on any allocated node that can read the
shared snapshot directory.

The viewer follows the newest atomically completed `.nbsnap` file while the
simulation is running. A successful simulator run publishes a completion
marker after its final frame; the viewer then automatically loops the complete
run and enables its timeline, play/pause button, and speed selector.

Build the host-only viewer on the cluster with:

```bash
cmake -S . -B build/browser-viewer -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DNBODY_ENABLE_CUDA=OFF \
  -DNBODY_BUILD_SIMULATION=OFF \
  -DNBODY_BUILD_VIEWER=OFF \
  -DNBODY_BUILD_STREAMING=ON
cmake --build build/browser-viewer --target nbody_browser_viewer --parallel
```

Pass the frame directory when launching it. The process prints every detected
node address that can be opened in a browser:

```bash
./build/browser-viewer/nbody_browser_viewer \
  --frames-dir /shared/path/to/run-frames \
  --bind 0.0.0.0 \
  --port 8080
```

The convenience wrapper builds the viewer when necessary:

```bash
./scripts/run-browser-viewer.sh /shared/path/to/run-frames --port 8080
```

The browser viewer must retain its frames, so use simulator `--stream-dir`
without `--cluster`. The latter starts the transfer server, which intentionally
deletes each frame after a download client acknowledges it. A typical batch or
interactive allocation starts both processes against the same directory:

```bash
RUN_DIR="${SCRATCH:-$PWD}/nbody-browser-${PBS_JOBID:-manual}"
mkdir -p "$RUN_DIR"

./build/simulation/nbody \
  --input data/two_body.bin \
  --time-step 0.04 \
  --steps 100000 \
  --stream-dir "$RUN_DIR" \
  --sample-rate 15 &

./build/browser-viewer/nbody_browser_viewer \
  --frames-dir "$RUN_DIR" \
  --bind 0.0.0.0 \
  --port 8080
```

The wildcard bind makes the port reachable wherever the cluster firewall
allows it and has no built-in authentication. For an SSH tunnel, bind only to
loopback and open the printed local port after forwarding it:

```bash
./build/browser-viewer/nbody_browser_viewer \
  --frames-dir "$RUN_DIR" --bind 127.0.0.1 --port 8080
```

Browser controls are:

- drag: orbit/rotate the system;
- Shift+drag or right-drag: shift the focus in the view plane;
- arrow keys: pan horizontally or vertically;
- vertical scrolling: pan in the same mirrored direction as the Metal viewer;
- Control+scroll or Control+Plus/Minus: zoom;
- `R` or **Reset view**: fit the current frame;
- Space, timeline, and speed selector: completed-run playback controls.

## Headless cluster frame streaming

The streaming path consists of three pieces:

- `nbody` samples completed CUDA particle states and atomically publishes
  typed position frames into a durable spool directory;
- `nbody_stream_server` sends the ordered spool backlog over TCP and removes a
  frame only after the client acknowledges that exact complete file;
- `nbody_stream_client` writes each download using a `.part` suffix, renames it
  atomically when complete, and then acknowledges it. Follow mode reconnects
  after network interruptions.

The sampler always writes the initial and final states. Between them, it emits
at most one frame every `1 / sample-rate` wall-clock seconds and only after a
simulation step completes. It never slows the CUDA loop merely to manufacture
the requested number of frames. By default, each frame contains an evenly
spaced deterministic sample of at most 100,000 particle indices. Set
`--stream-max-particles 0` to include every particle.

The simulator's complete streaming options are:

- `--cluster`: enable frames and start a child `nbody_stream_server`;
- `--stream-dir DIR`: select the durable spool and enable frames without, by
  itself, starting the server; default `nbody-stream` when `--cluster` is used;
- `--sample-rate HZ`: positive maximum wall-clock sampling rate; default `60`;
- `--stream-max-particles N`: non-negative display sample limit; default
  `100000`, while `0` includes every particle;
- `--stream-bind ADDRESS`: child-server bind address; default `127.0.0.1`;
- `--stream-port PORT`: child-server TCP port from 1 through 65535; default
  `4747`;
- `--stream-server FILE`: override the child server executable. By default the
  simulator looks for `nbody_stream_server` beside its own executable.

### Build the local download client

The client does not require CUDA:

```bash
cmake -S . -B build/client -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DNBODY_ENABLE_CUDA=OFF \
  -DNBODY_BUILD_SIMULATION=OFF \
  -DNBODY_BUILD_VIEWER=OFF \
  -DNBODY_BUILD_STREAMING=ON
cmake --build build/client --target nbody_stream_client --parallel
```

### Run the simulator and server on the cluster

Choose a spool directory that survives for as long as the backlog is needed:

```bash
STREAM_DIR="${SCRATCH:-$PWD}/nbody-frames-${SLURM_JOB_ID:-manual}"

./build/simulation/nbody \
  --input data/two_body.bin \
  --time-step 0.04 \
  --steps 100000 \
  --cluster \
  --stream-dir "$STREAM_DIR" \
  --sample-rate 60 \
  --stream-max-particles 100000 \
  --stream-bind 127.0.0.1 \
  --stream-port 4747
```

`--cluster` enables frame creation and starts the server built beside `nbody`.
`--stream-dir` without `--cluster` writes frames but does not start a server.
The automatically started server exits with the simulator, while every
unacknowledged `.nbsnap` file remains in the spool.

To serve a retained backlog after the simulation finishes, run on the cluster:

```bash
./build/simulation/nbody_stream_server \
  --spool-dir "$STREAM_DIR" \
  --bind 127.0.0.1 \
  --port 4747
```

Use `./build/simulation/nbody_stream_server --help` to see its optional
`--poll-ms` setting.

The standalone server accepts `--spool-dir DIR`, `--bind ADDRESS`,
`--port PORT`, and a positive `--poll-ms MS` no greater than 60000. Its defaults
are `nbody-stream`, `127.0.0.1`, `4747`, and `100` ms respectively.

### Forward the server port

The default loopback bind exposes no unauthenticated cluster port. Keep the
server terminal running and create one of these tunnels from the local Mac.

When the compute node is directly reachable:

```bash
ssh -N -L 4747:127.0.0.1:4747 user@compute-node
```

When the cluster requires a login-node jump host:

```bash
ssh -N -J user@login-node \
  -L 4747:127.0.0.1:4747 \
  user@compute-node
```

Replace the hostnames and usernames with the values assigned by the cluster.

### Download frames locally

In a second local terminal, run:

```bash
./build/client/nbody_stream_client \
  --host 127.0.0.1 \
  --port 4747 \
  --output-dir received-frames
```

Follow mode is the default and continues until interrupted with Control+C. Add
`--once` to download only the backlog available at connection time and exit:

```bash
./build/client/nbody_stream_client \
  --host 127.0.0.1 \
  --port 4747 \
  --output-dir received-frames \
  --once
```

Use `./build/client/nbody_stream_client --help` for the complete client options,
including `--reconnect-ms`.

The client accepts `--host ADDRESS`, `--port PORT`, `--output-dir DIR`,
`--once`, and a positive `--reconnect-ms MS` no greater than 60000. Its raw
defaults are `127.0.0.1`, `4747`, `nbody-frames`, follow mode, and `1000` ms
respectively.

Direct remote access is possible by binding the server to `0.0.0.0`, but this
transport has no built-in authentication or encryption. Only do so behind a
cluster firewall; SSH tunnelling is the normal configuration.

### Convenience wrappers

The wrappers configure and build their application when necessary:

```bash
# On the cluster, serve the current job's spool.
./scripts/run-stream-server.sh

# Locally, follow frames through an already-open SSH tunnel.
./scripts/run-stream-client.sh

# Locally, download the current backlog and exit.
./scripts/run-stream-client.sh --once
```

Shared overrides include `NBODY_STREAM_HOST`, `NBODY_STREAM_PORT`, and
`NBODY_SKIP_BUILD=1`. Server overrides include `NBODY_STREAM_SPOOL_DIR`,
`NBODY_STREAM_BIND`, `NBODY_STREAM_POLL_MS`, `NBODY_SERVER_BUILD_DIR`, and
`NBODY_SERVER_EXECUTABLE`. Client overrides include `NBODY_FRAME_OUTPUT_DIR`,
`NBODY_CLIENT_BUILD_DIR`, and `NBODY_CLIENT_EXECUTABLE`.

The raw client defaults to `nbody-frames` relative to its working directory.
The client wrapper defaults to `received-frames` in the project directory. The
server wrapper uses `$SCRATCH/nbody-frames-$SLURM_JOB_ID` when both cluster
variables exist, with safe local/manual fallbacks otherwise.

### Stream frame format and viewer compatibility

Each current `.nbsnap` file is little-endian and independently decodable. Its
128-byte version-3 header contains the `NBSNAP01` magic, format version,
sequence number, simulation step, simulated time, source and sample particle
counts, scalar/component sizes, payload size, and double-precision minimum and
maximum bounds for each position axis. The payload contains interleaved
unsigned 16-bit `x, y, z` values followed by one `uint8` particle type per
sampled body. A consumer reconstructs component `c` with:

```text
position[c] = minimum[c]
            + quantized[c] * (maximum[c] - minimum[c]) / 65535
```

An axis whose minimum equals its maximum reconstructs to that single value.
Version 3 uses 7 bytes per sampled particle: 6 position bytes and one stable
body-type byte. Quantization runs on the GPU before the payload is copied to
the host; the simulator's full double-precision state remains unchanged in
memory. Frames do not depend on earlier frames, so backlog downloads,
reconnection, and browser seeking need no keyframe.

The frame reader also accepts retained version-2 position-only frames with a
120-byte header and retained version-1 files with a 72-byte header and
interleaved float32 positions. Their missing body types are exposed as Unknown.

Filenames include a unique run ID, so a later run cannot overwrite an older
offline backlog.

The current Metal viewer cannot read `.nbsnap` files: it expects a full planar
particle file with masses, double-precision positions and velocities, and
optional particle types. The streaming client is therefore currently useful
for durable transfer, archival, tests, or a separate frame consumer—not for
direct playback in `nbody_viewer`.

## Particle file format

The Python generators write this little-endian structure-of-arrays layout:

1. one `uint64` particle count;
2. all `mass` values as float64;
3. all `x`, then `y`, then `z` positions as float64;
4. all `vx`, then `vy`, then `vz` velocities as float64;
5. one `uint8` particle type per particle.

Stable particle-type values are `0` unknown, `1` star, `2` planet, `3` moon,
and `4` asteroid. The production loader and Metal viewer also accept legacy
files with a `uint32` count or without the final type block. The file size must
exactly match one of those layouts.

The full particle `.bin` format and the typed display-sample `.nbsnap` format
are intentionally different and are not interchangeable.

## Generate synthetic data

`data/generate_galaxy.py` creates loader-ready synthetic datasets. Run
`python3 data/generate_galaxy.py --help` for its parser-generated summary.

Useful options are:

- `--galaxy globular|spiral|two-body`: required dataset type. The current
  `globular` and `spiral` paths use the same Gaussian random-data generator;
  their names affect the default output filename, not the coordinates;
- `--shape natural|heart|smile`: accepted for non-two-body data and used in the
  default output filename, but currently does not transform the coordinates;
  default `natural`;
- `-n, --num-particles N`: particle count; default 100,000 for galaxy modes and
  exactly 2 for two-body mode;
- `--num-arms N`: accepted positive integer; default `4`. It currently does
  not alter the generated coordinates;
- `--mass VALUE`: mass per particle; defaults to `1e30` kg for galaxy modes and
  `1e10` kg for two-body mode;
- `--separation VALUE`: two-body centre-to-centre distance; default `1.0`;
- `-o, --output PATH`: output path; when omitted, the script creates a named
  file under `data/`;
- `--seed N`: reproducible random seed; default `12345`.

Examples:

```bash
python3 data/generate_galaxy.py --galaxy globular -n 100000 -o data/globular_cloud.bin
python3 data/generate_galaxy.py --galaxy spiral -n 5000000 --seed 7 -o data/random_cloud_5M.bin
python3 data/generate_galaxy.py --galaxy two-body -o data/two_body.bin
```

Two-body mode creates equal masses in a circular orbit about their shared
centre of mass. It prints a suggested timestep for 128 steps per orbit; use
that printed value for the most precise matching run:

```bash
./build/simulation/nbody \
  --input data/two_body.bin \
  --time-step <printed-suggested-timestep> \
  --steps 128 \
  --theta 0.5 \
  --softening 0
```

Replace the placeholder, including its angle brackets, with the printed numeric
value. Use a writable relative output such as `data/file.bin`; `/data/file.bin`
refers to a root-level directory and will normally fail without permission.

The default random galaxy datasets are primarily visual/load-test inputs. Their
default scales are not intended as a calibrated astrophysical model. Both the
random-cloud and two-body modes currently label their generated particles as
stars, so the viewer renders them in the star colour.

## Download Solar-System data

`data/download_solar_system.py` builds two loader-ready datasets at selected TDB
epochs. It combines barycentric ICRF vectors for the Sun, planet centres, and
JPL-listed planetary moons with the IAU Minor Planet Center asteroid catalogue.
The command requires NumPy, internet access, substantial download time for the
full catalogue, and enough free disk space for its cache and outputs.

Run `python3 data/download_solar_system.py --help` for all download, cache,
catalogue-limit, physical-assumption, and output options.

```bash
python3 data/download_solar_system.py \
  --epoch-a 2026-08-26T00:00:00 \
  --epoch-b 2026-09-25T00:00:00 \
  --output-dir data/solar_system
```

The output directory contains:

- `solar_system_epoch_a.bin` and `solar_system_epoch_b.bin`, with identical
  body ordering and the full particle layout documented above;
- `solar_system_objects.csv.gz`, mapping every index to its source object and
  mass model;
- `solar_system_metadata.json`, recording epochs, units, assumptions, sources,
  counts, and skipped objects.

Downloads and individual HORIZONS responses are cached under
`.cache/solar_system`, so an interrupted build can resume. Use `--refresh` to
update source files, `--offline` to require cached data, and `--overwrite` to
replace existing outputs. For a quick pipeline check, add
`--max-asteroids 1000` before processing the full catalogue.

Major-body positions and velocities are HORIZONS geometric states. MPC
asteroids are propagated independently from heliocentric osculating elements
with a two-body Kepler model, rotated from the J2000 ecliptic to ICRF, and
translated using the HORIZONS barycentric Sun state. Known JPL `GM` values are
converted to mass with the same `G` as the simulator. Most asteroid masses are
estimated from absolute magnitude using configurable albedo and density
assumptions. Moons without a published JPL `GM` are retained as massless test
particles. These approximations are suitable for cluster-scale application and
convergence testing, but the asteroid dataset is not a high-precision HORIZONS
reference trajectory for every object.

## Tests

Tests are divided into CPU/CUDA unit tests and CUDA integration tests. CMake
builds the production libraries once and links the GoogleTest executables
against them.

Run the CPU tests on macOS with a C++23 compiler. This project is validated with
Homebrew LLVM 18; adjust or omit `CMAKE_CXX_COMPILER` if an equivalent compiler
is already selected:

```bash
cmake -S . -B build/cpu-tests -G Ninja \
  -DBUILD_TESTING=ON \
  -DNBODY_ENABLE_CUDA=OFF \
  -DNBODY_BUILD_SIMULATION=OFF \
  -DNBODY_BUILD_VIEWER=OFF \
  -DNBODY_BUILD_STREAMING=ON \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@18/bin/clang++
cmake --build build/cpu-tests --parallel
ctest --test-dir build/cpu-tests --output-on-failure
```

Run all CUDA tests on a Linux/WSL machine with the CUDA toolkit and a compatible
GPU:

```bash
cmake -S . -B build/cuda-tests -G Ninja \
  -DBUILD_TESTING=ON \
  -DNBODY_ENABLE_CUDA=ON \
  -DNBODY_BUILD_SIMULATION=ON \
  -DNBODY_BUILD_VIEWER=OFF \
  -DNBODY_BUILD_STREAMING=ON
cmake --build build/cuda-tests --parallel
ctest --test-dir build/cuda-tests -L cuda --output-on-failure
```

Useful filters are:

```bash
ctest --test-dir build/cuda-tests -L unit --output-on-failure
ctest --test-dir build/cuda-tests -L integration --output-on-failure
ctest --test-dir build/cuda-tests -R TreeBuilder --output-on-failure
ctest --test-dir build/cuda-tests -R SolarSystemEphemerisTest --output-on-failure
```

Compute Sanitizer is intentionally separate from the fast test run:

```bash
tests/scripts/run_cuda_sanitizers.sh
RUN_RACECHECK=1 tests/scripts/run_cuda_sanitizers.sh
```

The CUDA integration suite includes `SolarSystemEphemerisTest`. Its compiled
fixture writer creates two compact Sun-and-planets datasets at runtime for
2026-08-26 and 2026-08-27 TDB without Python, network access, or committed
binary data. Epoch A initializes the production simulation; after one simulated
day, the resulting 3D positions and velocities are checked against epoch B with
per-body tolerances.

## Numerical-method background

For a second-order equation $\ddot{x}(t)=A(x(t))$, Verlet-style integration
uses previous and current positions to advance the state. Compared with forward
Euler, the leapfrog/Verlet family has better long-term behaviour for many
Hamiltonian systems, including time symmetry and a second-order global error
under the usual smoothness assumptions. The production implementation uses a
shared leapfrog timestep rather than the obsolete plotting workflow that was
previously described in this README.
