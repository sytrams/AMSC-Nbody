# NBody
In an N-body problem, we need to find the **positions** and **velocities** of a **collection of interacting particles** over a period of time. An n-body solver is a program that finds the solution to an n-body problem by simulating the behavior of the particles. The input to the problem is the **mass**, **position**, and **velocity** of each particle at the start of the simulation, and the output is typically the **position** and **velocity** of each particle at a sequence of user-specified times.
We’ll use **Newton’s second law of motion** and his **law of universal gravitation** to determine the positions and velocities. 
If particle q has position $s_q(t)$ at time t , and particle k has position $s_k(t)$, then the force on particle q exerted by particle k is given by:

$$
f_{qk}(t) = -\frac{G m_q m_k}{|s_q(t) - s_k(t)|^3} \cdot [s_q(t) - s_k(t)]
$$

**Verlet integration** is a numerical method used to integrate Newton's equations of motion. It is frequently used to calculate trajectories of particles in molecular dynamics simulations and computer graphics. The Verlet integrator **provides good numerical stability**, as well as other properties that are important in physical systems such as time reversibility and preservation of the symplectic form on phase space.
For a second-order differential equation of the type $\ddot{x}(t)=A(x(t))$ with initial conditions $x(t_{0})=x_{0}$ and $\dot{x}(t_{0})=v_{0}$ an approximate numerical solution $x_{n} \approx x(t_{n})$ at the times $t_{n}=t_{0}+n \Delta t$ with step size $\Delta t > 0$ can be obtained by the following method:

**Initialization:**

$$
x_1 = x_0 + v_0 \Delta t + \frac{1}{2} A(x_0) \Delta t^2
$$

**Iteration for $n = 1, 2, \dots$:**

$$
x_{n+1} = 2x_n - x_{n-1} + A(x_n) \Delta t^2
$$


## solar_system.txt

<div align="center">
  
![Schema Verlet](img/planets_animation.gif)

</div>

The file to be provided within the code must have the following structure to be read consistently and correctly:
the first line contains, in order:
- **DIM** dimension of the space in which the problem is to be represented and evaluated,
- the **time variation** **$\Delta t$**, and
- the **total time interval** over which the problem is to be studied.
In our study we chose a simpler case of 2 dimension plane and a variation time delta_t of 1 day(86400 seconds) and 10 years(3.155692608e8 seconds) for the total amount of time.

After that, the following lines list the various bodies, indicating:
- their **name**
- their **mass**
- the components of the initial **position** (2 or 3 depending on the dimension chosen to represent), for which we decided to use the average distance from the Sun
- the components of the initial **tangential velocity** (always 2 or 3), for which we decided to use the average velocity of each planet.

In our case, for an initial analysis, we began by studying the case in 2 dimensions over a total time period of 10 years (total_period = 10(year)*365.2422(days per year)*24(hours per day)*60(minutes per hours)*60(seconds per minute) = 315'569'260.8 seconds) using a delta_t of 1 day = 86,400 seconds (24*60*60).


<div align="center">
 
![Schema Verlet](img/trajectory.png)


</div>

## Simulation executable

The CUDA simulation executable is opt-in and requires an NVIDIA CUDA toolkit.
Configure and build it with:

```bash
cmake -S . -B build/simulation -G Ninja \
  -DBUILD_TESTING=OFF \
  -DNBODY_ENABLE_CUDA=ON \
  -DNBODY_BUILD_SIMULATION=ON \
  -DNBODY_BUILD_VIEWER=OFF
cmake --build build/simulation --target nbody --parallel
```

Run a fixed number of shared leapfrog timesteps with:

```bash
./build/simulation/nbody \
  --input <initial-conditions.bin> \
  --time-step <dt> \
  --steps <count> \
  --theta 0.5 \
  --softening <length>
```

Use `./build/simulation/nbody --help` for the complete command-line help.
The input must use the planar binary particle format documented below. The
executable advances particles in memory unless `--stream-dir` or `--cluster`
enables graphical snapshots.

## Headless cluster frame streaming

The graphical output path consists of three pieces:

- `nbody` samples completed particle states at a maximum of 60 wall-clock Hz
  and atomically publishes position frames into a durable spool directory.
- `nbody_stream_server` watches that directory and streams its ordered backlog
  over TCP. It removes a frame only after the client acknowledges the exact
  complete file.
- `nbody_stream_client` saves each download with a `.part` suffix, atomically
  renames it when complete, and then acknowledges it. Follow mode reconnects
  after a network interruption.

The sampler always writes the initial and final states. Between them it emits
at most one frame every `1 / sample-rate` wall-clock seconds and only after a
simulation step completes. It never delays the CUDA loop to manufacture 60
frames when a step itself takes longer than 1/60 second. To keep large runs
viewable, each frame contains an evenly spaced, deterministic sample of at most
100,000 particle indices by default. Use `--stream-max-particles N` to change
that limit or `--stream-max-particles 0` to send every particle.

Streaming targets are built by default. A client-only build does not require
CUDA:

```bash
cmake -S . -B build/client -G Ninja \
  -DBUILD_TESTING=OFF \
  -DNBODY_ENABLE_CUDA=OFF \
  -DNBODY_BUILD_SIMULATION=OFF \
  -DNBODY_BUILD_VIEWER=OFF \
  -DNBODY_BUILD_STREAMING=ON
cmake --build build/client --target nbody_stream_client --parallel
```

On the cluster, `--cluster` enables frames and starts the server that was built
beside the simulator:

```bash
./build/simulation/nbody \
  --input data/initial.bin \
  --time-step 0.001 \
  --steps 100000 \
  --cluster \
  --stream-dir "$SCRATCH/nbody-frames-$SLURM_JOB_ID" \
  --sample-rate 60 \
  --stream-max-particles 100000 \
  --stream-bind 127.0.0.1 \
  --stream-port 4747
```

The automatically started server exits with the simulator, but every
unacknowledged `.nbsnap` file stays in the spool. To retrieve a backlog after
the simulation has ended, start the server explicitly:

```bash
./build/simulation/nbody_stream_server \
  --spool-dir "$SCRATCH/nbody-frames-$SLURM_JOB_ID" \
  --bind 127.0.0.1 \
  --port 4747
```

The default loopback binding deliberately exposes no unauthenticated cluster
port. Forward it through SSH (using the login node as a jump host when the
compute node requires one), then run the client locally:

```bash
ssh -N -L 4747:127.0.0.1:4747 user@compute-node

./build/client/nbody_stream_client \
  --host 127.0.0.1 \
  --port 4747 \
  --output-dir nbody-frames
```

The client follows new frames until interrupted. Add `--once` to download only
the backlog present at connection time and exit. Direct remote access is also
possible with `--stream-bind 0.0.0.0`, but it should be used only behind a
cluster firewall because this initial transport intentionally relies on SSH
for authentication and encryption.

Each `.nbsnap` file is little-endian and contains a 72-byte header followed by
interleaved float32 `x, y, z` values. The header contains the `NBSNAP01` magic,
format version, sequence number, simulation step, simulated time, particle
source/sample counts, scalar/component sizes, and payload byte count. Only
positions are sent to keep graphical bandwidth at 12 bytes per sampled particle
per frame; the simulator's full double-precision state is unchanged. Filenames
include a unique run id, so an older offline backlog cannot be overwritten by a
later simulation.

## Tests

Tests live under `tests/` and are divided into CPU/CUDA unit tests and
CUDA integration tests. CMake builds the production libraries once and
links the GoogleTest executables against them.

Run the CPU tests on macOS:

```bash
cmake -S . -B build/cpu-tests \
  -G Ninja \
  -DBUILD_TESTING=ON \
  -DNBODY_ENABLE_CUDA=OFF \
  -DNBODY_BUILD_VIEWER=OFF \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@18/bin/clang++
cmake --build build/cpu-tests --parallel
ctest --test-dir build/cpu-tests --output-on-failure
```

The equivalent convenience command is:

```bash
make test
```

Run all CUDA tests on a Linux/WSL machine with the CUDA toolkit:

```bash
cmake -S . -B build/cuda-tests \
  -G Ninja \
  -DBUILD_TESTING=ON \
  -DNBODY_ENABLE_CUDA=ON \
  -DNBODY_BUILD_SIMULATION=ON \
  -DNBODY_BUILD_VIEWER=OFF
cmake --build build/cuda-tests --parallel
ctest --test-dir build/cuda-tests -L cuda --output-on-failure
```

Useful filters include:

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

The macOS Metal viewer is a separate executable and never opens as a
side effect of running tests:

```bash
cmake -S . -B build/viewer -G Ninja \
  -DBUILD_TESTING=OFF \
  -DNBODY_BUILD_VIEWER=ON
cmake --build build/viewer --target nbody_viewer
./build/viewer/nbody_viewer -f data/test_spiral.bin
```

## Generate Galaxy Data
The script [data/generate_galaxy.py](/Users/wedoi/Documents/AMSC-nbody/AMSC-Nbody/data/generate_galaxy.py:1) creates synthetic binary datasets for the galaxy viewer and the large-particle loader.

Run it from the project root with:
```bash
python3 data/generate_galaxy.py --galaxy spiral -n 100000 -o data/milky_way_like.bin
```

Available options:
- ```--galaxy globular|spiral|two-body``` selects the initial-condition type.
- ```--shape natural|heart|smile``` changes the projected XY shape. The default is ```natural```.
- ```--num-arms N``` sets the number of spiral arms when ```--galaxy spiral``` is used. The default is ```4```.
- ```-n``` or ```--num-particles``` sets the particle count.
- ```--mass VALUE``` sets the mass of each particle. The two-body default is ```1e10``` kg.
- ```--separation VALUE``` sets the distance between the bodies when ```--galaxy two-body``` is used.
- ```-o``` or ```--output``` sets the output binary file path.
- ```--seed``` makes the generated dataset reproducible.

Examples:
```bash
python3 data/generate_galaxy.py --galaxy spiral --num-arms 3 -n 5000000 -o data/milky_way_like_5M.bin
python3 data/generate_galaxy.py --galaxy globular --shape heart -n 100000 -o data/globular_heart.bin
python3 data/generate_galaxy.py --galaxy spiral --shape smile -n 100000 -o data/spiral_smile.bin
python3 data/generate_galaxy.py --galaxy two-body -o data/two_body.bin
```

The two-body mode generates two equal masses in a circular orbit around their
shared center of mass. It prints the orbital period and a suggested timestep
for 128 steps per orbit. Run the resulting file with, for example:

```bash
./build/simulation/nbody \
  --input data/two_body.bin \
  --time-step <suggested-timestep> \
  --steps 128 \
  --theta 0.5 \
  --softening 0
```

Binary format:
- the first 8 bytes are the number of particles stored as little-endian ```uint64```
- then all ```mass``` components
- then all ```x``` coordinates
- then all ```y``` coordinates
- then all ```z``` coordinates
- then all ```vx``` coordinates
- then all ```vy``` coordinates
- then all ```vz``` coordinates

Each coordinate and velocity component is written as little-endian ```double```.

Use a writable output path such as ```data/file.bin```. A path like ```/data/file.bin``` points to a root-level directory and will usually fail with a permission error.

## Download Solar-System Data

The script [data/download_solar_system.py](/Users/wedoi/Documents/AMSC-nbody/AMSC-Nbody/data/download_solar_system.py:1) builds two loader-ready datasets at user-selected TDB epochs. It combines barycentric ICRF vectors for the Sun, planet centers, and all planetary moons listed by JPL HORIZONS with the full IAU Minor Planet Center asteroid catalogue.

```bash
python3 data/download_solar_system.py \
  --epoch-a 2026-08-26T00:00:00 \
  --epoch-b 2026-09-25T00:00:00 \
  --output-dir data/solar_system
```

The output directory contains:

- `solar_system_epoch_a.bin` and `solar_system_epoch_b.bin`, with identical body ordering and the binary layout described above;
- `solar_system_objects.csv.gz`, which maps every particle index to its source object and mass model;
- `solar_system_metadata.json`, which records epochs, units, assumptions, sources, counts, and skipped objects.

Downloads and individual HORIZONS responses are cached under `.cache/solar_system`, so rerunning an interrupted build resumes from the cache. Use `--refresh` to update the source files, `--offline` to require cached data, and `--overwrite` to replace existing outputs. For a quick pipeline check before processing the million-scale catalogue, add `--max-asteroids 1000`.

Major-body positions and velocities are HORIZONS geometric states. MPC asteroids are propagated independently from their heliocentric osculating elements with a two-body Kepler model, rotated from the J2000 ecliptic to ICRF, and translated using the HORIZONS barycentric Sun state. Known JPL `GM` values are converted to mass using the same `G` as the simulator. Most asteroid masses are not measured, so the script estimates them from absolute magnitude using configurable albedo and density assumptions. Moons without a published JPL `GM` are retained as massless test particles. These approximations make the files suitable for cluster-scale application and convergence testing, but the second dataset is not a high-precision HORIZONS reference trajectory for every asteroid.

The CUDA integration suite contains `SolarSystemEphemerisTest`. Its compiled
C++ fixture writer creates two compact Sun-and-planets datasets at runtime for
2026-08-26 and 2026-08-27 TDB, without Python, network access, or committed
binary data. Epoch A initializes the production simulation; after one simulated
day, all resulting 3D positions and velocities are checked against epoch B.
Per-body tolerances account for the moons deliberately omitted from this test.

## External tool
We have and external script writing in python to export an image of the plot about our simulation.

In order to run it you have to execute
```
python3 plt_trajectory.py
```
from the ```src``` directory

The image can be seen as ```/build/trajectory.png```


# The general issue we had
In our initial study with the Euler method we found a problem when representing the trajectories as they did not seem to be precise and to respect the correct progression of the planets.
For this reason we decided to adopt and study our case by applying the Verlet integration, which is a numerical method used to integrate Newton's equations of motion.
The Verlet integrator provides good numerical stability, as well as other properties that are important in physical systems such as time reversibility and preservation of the symplectic form on phase space (maintain the fundamental geometric structure of phase space intact, avoiding energy drift and ensuring long-term stability), at no significant additional computational cost over the simple Euler method.
Euler’s error tends to accumulate dramatically, often causing solutions to “blow up.”
The Verlet method, thanks to its time-symmetric structure, is much more stable, and moreover it has a smaller overall error:

•	Euler: local error of order 1 → global error O(Δt)

•	Verlet: local error of order 2 → global error O(Δt²)
