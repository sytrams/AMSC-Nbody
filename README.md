# AMSC N-Body
Silvestri Martina e Storti Edoardo

## CUDA Barnes-Hut N-Body Simulation

This project implements a three-dimensional gravitational N-body simulator designed for GPU execution with CUDA. In this design C++ is used as a CUDA kernel manager exploiting its containers and safe pointers in order to safely manage the host-device memory communication. To give CUDA an higher level of abstraction without losing performance it was essential to take advantage of the C++ classes and methods. Namespaces were used to create private environments in header files and source codes to achieve safe access throughout the whole project.

The objective is to simulate systems containing a large number of gravitationally interacting particles while avoiding the $(O(N^2)$ computational cost of direct all-pairs force evaluation. The implementation therefore combines the Barnes-Hut approximation with a GPU-oriented spatial data structure based on Morton codes, radix sorting, a Karras radix tree, and a sparse octree.

The final simulation pipeline includes:

- three-dimensional particle dynamics;
- double-precision particle state and force computation;
- CUDA execution;
- Morton spatial ordering;
- GPU radix sorting with CUB;
- Karras-style radix-tree construction;
- conversion to a sparse octree;
- bottom-up computation of node masses and centers of mass;
- Barnes-Hut gravitational acceleration;
- stackless octree traversal;
- Morton-ordered target traversal and particle storage;
- leapfrog time integration;
- configurable Barnes-Hut opening angle and gravitational softening;
- CPU and CUDA unit tests;
- CUDA integration and physical-accuracy tests;
- Compute Sanitizer validation;
- headless frame streaming for cluster execution;
- a playback-capable Metal viewer for live and archived simulations;
- a cluster-hosted browser viewer for live and completed frame directories.

---

# 1. Physical model

For a system containing $N$ particles, particle $i$ has mass $m_i$, position

$$
\mathbf{r}_i =
\begin{bmatrix}
x_i\\
y_i\\
z_i
\end{bmatrix},
$$

and velocity

$$
\mathbf{v}_i =
\begin{bmatrix}
v_{x,i}\\
v_{y,i}\\
v_{z,i}
\end{bmatrix}.
$$

According to Newton's law of universal gravitation, the acceleration generated on particle $i$ by particle $j$ is

$$
\mathbf{a}_{ij} =
Gm_j
\frac{\mathbf{r}_j-\mathbf{r}_i}
{\left(\|\mathbf{r}_j-\mathbf{r}_i\|^2+\varepsilon^2\right)^{3/2}},
$$


where:

- $G$ is the gravitational constant;
- $\epsilon$ is the gravitational softening parameter.

The total acceleration of particle $i$ is therefore

$$
\mathbf{a}_i =
\sum_{\substack{j=0\\j\neq i}}^{N-1}
\mathbf{a}_{ij}.
$$

The softening term prevents singular accelerations for particles at extremely small separation and can be configured from the command line.

Setting the softening to zero is also supported, provided that the simulated configuration does not generate singular interactions.

A direct evaluation of this expression requires every particle to interact with every other particle, resulting in

$$
O(N^2)
$$

work per simulation step. This becomes prohibitively expensive for large particle counts.

The main purpose of the Barnes-Hut algorithm used in this project is to reduce this computational cost.

---

# 2. Barnes-Hut approximation

Barnes-Hut reduces the number of gravitational interactions by grouping sufficiently distant particles.

The three-dimensional simulation domain is represented by an octree. Each internal node represents a cubic region of space and stores:

- its geometric center;
- its half-size;
- its total mass;
- its center of mass;
- its parent;
- references to up to eight children.

Occupied leaves additionally identify the range of particles contained in the corresponding spatial cell.

When evaluating the acceleration of one target particle, an internal octree node can either be:

1. **opened**, causing its children to be inspected; or
2. **accepted**, replacing all particles below that node with a single point mass located at the node center of mass.

Leaves are always evaluated exactly.

The approximation is controlled by the Barnes-Hut opening parameter

$$
\theta.
$$

Smaller values of $\theta$ cause more nodes to be opened and therefore increase accuracy at the cost of additional computation.

Larger values allow more aggressive approximation and reduce execution time.

The current implementation also accounts for the displacement between the geometric center of a cell and its actual center of mass. For a node of side length $s$, target-to-center-of-mass distance $d$, and center-to-center-of-mass displacement $\delta$, the node is accepted only when

$$
d > \frac{s}{\theta}+\delta.
$$

This criterion is more conservative for strongly asymmetric cells than the basic $s/d < \theta$ Barnes-Hut criterion.

A node containing the target particle is never approximated as a single point mass.

With $\theta = 0$, no internal node is accepted and the traversal behaves as an exact tree-based N-body calculation.

---

# 3. Time integration

Particle trajectories are advanced with a leapfrog kick-drift-kick integrator.

Leapfrog was selected because it is a second-order symplectic method and provides substantially better long-term behavior for gravitational systems than a first-order explicit method such as forward Euler.

Given particle position $r_n$, velocity $v_n$, acceleration $a_n$, and timestep $\Delta t$, one complete step is

$$
\mathbf{v}_{n+\frac12} =
\mathbf{v}_n
+
\frac{\Delta t}{2}\mathbf{a}_n,
$$

$$
\mathbf{r}_{n+1} =
\mathbf{r}_n
+
\Delta t\,\mathbf{v}_{n+\frac12},
$$

followed by reconstruction of the spatial structure and evaluation of the new acceleration

$$
\mathbf{a}_{n+1} = \mathbf{a}(\mathbf{r}_{n+1}),
$$



and finally

$$
\mathbf{v}_{n+1} =
\mathbf{v}_{n+\frac12}
+
\frac{\Delta t}{2}\mathbf{a}_{n+1}.
$$

The implementation follows this structure directly.

After the drift modifies particle positions, the Morton ordering and octree are rebuilt before the next Barnes-Hut force computation.

---

# 4. GPU spatial pipeline

The spatial structure is reconstructed every timestep because particle positions change continuously.

The production pipeline is:

```text
Particle positions
        |
        v
Global bounding box
        |
        v
Coordinate normalization
        |
        v
Morton key generation
        |
        v
CUB radix sort (key, particle index)
        |
        v
Morton leaf groups
        |
        v
Karras radix tree
        |
        v
Radix-tree to octree conversion
        |
        v
Sparse octree topology
        |
        v
Octree geometry
        |
        v
Mass / center-of-mass propagation
        |
        v
Barnes-Hut traversal
        |
        v
Particle accelerations
        |
        v
Leapfrog integration
```

Each stage has dedicated CUDA kernels and corresponding unit or integration tests.

---

# 5. Global bounding box

Morton encoding requires particle coordinates to be mapped into a finite integer grid.

At every rebuild, the global minimum and maximum particle coordinates are computed on the GPU.

The resulting bounds define a cubic simulation region used to normalize all three spatial coordinates consistently.

Using a cube instead of three independently scaled dimensions preserves the spatial meaning of the Morton representation.

---

# 6. Morton encoding

Normalized particle coordinates are quantized onto a three-dimensional integer grid.

The implementation uses 10 bits per coordinate, producing a 30-bit Morton code inside a 32-bit integer.

For coordinates

$$
(x_q,y_q,z_q),
$$

their bits are interleaved to generate a Morton key.

Morton ordering provides a one-dimensional representation that approximately preserves three-dimensional spatial locality: particles close in space tend to have similar Morton keys.

This property is fundamental to several later optimizations.

---

# 7. GPU radix sorting

Morton keys are sorted with

```text
cub::DeviceRadixSort::SortPairs
```

using pairs consisting of

```text
(Morton key, original particle index).
```

The particle index permutation is preserved throughout tree construction.

This is essential because sorting only the Morton keys would destroy the relation between a tree leaf and the physical particle represented by that key.

The sorted particle indices therefore provide both:

- the spatial ordering used to build the tree;
- the mapping between Morton order and the canonical particle order.

---

# 8. Morton leaf groups

Multiple particles may share the same Morton key.

This can happen when different particles fall into the same quantized spatial cell or when particles have identical positions.

Instead of requiring every Morton key to be unique, consecutive equal keys are grouped into Morton leaf groups.

Each group stores the corresponding sorted particle range.

This makes duplicate Morton keys a supported input rather than a special failure case.

---

# 9. Karras radix tree

The sorted Morton groups are first organized into a binary radix tree following the parallel construction principles described by Karras.

For each internal node, the algorithm determines the range of Morton prefixes represented by the node and identifies the corresponding split.

The GPU construction uses prefix information derived from the sorted keys and creates the complete radix-tree topology in parallel.

Only the topology required for the later octree conversion is allocated in the simulation path. Physical quantities that are not used by the radix tree are intentionally omitted.

---

# 10. Sparse octree construction

The binary radix tree is subsequently converted into a sparse three-dimensional octree.

Unlike a dense octree, empty spatial cells are not allocated.

Each octree node stores only existing children, reducing memory consumption for sparse particle distributions.

The octree contains:

- eight child references per node;
- a parent reference;
- level and prefix information;
- leaf particle ranges;
- node geometry;
- physical mass and center-of-mass information.

The root always represents the complete simulation bounding cube.

---

# 11. Octree geometry

Once the topology is known, every octree node receives its geometric center and half-size.

The geometry is derived from:

- the global bounding box;
- the node level;
- the Morton/octant prefix represented by that node.

The geometry is required by the Barnes-Hut acceptance criterion and is independent from the physical center of mass.

---

# 12. Mass and center of mass

After topology and geometry are available, leaf masses and centers of mass are computed from the particles belonging to each leaf.

For a node containing particles with masses $m_i$ and positions $r_i$,

$$
M = \sum_i m_i
$$

and

$$
\mathbf{r}_{CM} =
\frac{1}{M}
\sum_i m_i\mathbf{r}_i.
$$

Internal-node values are then propagated bottom-up.

This computation is performed on the GPU.

Zero-mass particles are supported. A zero-total-mass node produces no gravitational acceleration and can be skipped during traversal.

---

# 13. Stackless Barnes-Hut traversal

An earlier version of the Barnes-Hut CUDA kernel used a private traversal stack for every CUDA thread.

For the maximum octree depth used by the project, this required an array equivalent to approximately

```text
int stack[71]
```

per thread.

Compiler inspection showed a 288-byte stack frame for the kernel.

The final implementation uses a stackless depth-first traversal instead.

Each octree node already stores:

- its parent;
- its children.

These links are sufficient to find:

- the first child of an opened node;
- the next sibling;
- the next ancestor containing an unvisited sibling.

The explicit per-thread stack was therefore removed.

The optimized kernel has no equivalent large traversal-stack allocation and avoids the associated local-memory pressure.

The traversal order was kept consistent with the previous implementation to preserve deterministic behavior.

---

# 14. Morton-ordered Barnes-Hut targets

A major source of CUDA inefficiency in a tree algorithm is warp divergence.

If adjacent CUDA threads process particles that are spatially unrelated, they are likely to traverse very different paths through the octree.

The simulation instead assigns target particles to threads in sorted Morton order.

Consequently, consecutive threads in a warp normally process spatially close particles.

This increases the probability that those threads make similar Barnes-Hut open/accept decisions during the upper levels of traversal.

The final acceleration is still written to the original particle index, so the rest of the simulator retains its canonical particle ordering.

---

# 15. Morton-ordered physical particle data

Morton ordering of the target threads improves traversal coherence, but an initial version still accessed particle coordinates through indirect indexing:

```text
sorted index
     |
     v
original particle index
     |
     v
x[index], y[index], z[index], mass[index]
```

Nsight Compute showed a large number of inefficient memory transactions and significant latency associated with the irregular traversal and particle access pattern.

The final implementation therefore maintains persistent simulation buffers containing

```text
mass
x
y
z
```

in Morton order specifically for Barnes-Hut evaluation.

After the Morton permutation is generated, a CUDA gather kernel fills these buffers.

Barnes-Hut can then access particles inside a Morton leaf directly through their sorted position, while acceleration output is still mapped back to the original particle index.

This preserves the canonical state required by the integrator while improving memory locality inside the force calculation.

The cost of the gather is included in the measured simulation step time.

---

# 16. Data layout and memory management

Particle and octree data use a structure-of-arrays layout.

Instead of storing

```text
Particle {
    x, y, z,
    vx, vy, vz,
    mass
}
```

as a single array of structures, individual quantities are stored in separate contiguous arrays.

This representation is suitable for CUDA because kernels often access one quantity for many particles simultaneously.

CUDA allocations owned by tree and octree structures are managed through RAII-style device buffers.

This reduces the number of raw owning pointers and ensures that device allocations are released correctly when structures are destroyed or when construction fails.

Spatial structures are staged before being committed to the active simulation state, preventing a failed rebuild from partially replacing a valid octree.

---

# 17. CUDA launch configuration

The project originally used a fixed CUDA architecture.

The current CMake configuration uses

```text
CMAKE_CUDA_ARCHITECTURES=native
```

when the caller does not explicitly provide an architecture.

This makes local builds target the installed GPU while still allowing cluster or CI builds to override the architecture.

Nsight Compute was also used to tune the Barnes-Hut block size.

For the RTX 3070 used during profiling, the Barnes-Hut kernel required approximately

```text
50 registers / thread
```

and a 256-thread block produced:

```text
Theoretical occupancy: 66.67 %
Achieved occupancy:    59.19 %
```

Reducing the block size to 128 threads increased the values to approximately

```text
Theoretical occupancy: 75.00 %
Achieved occupancy:    65.39 %
```

and produced a small but reproducible improvement in complete simulation time.

A 96-thread configuration was also tested and was slower, therefore 128 threads per block is used by the final Barnes-Hut kernel.

---

# 18. Performance analysis

Profiling was performed with NVIDIA Nsight Systems and Nsight Compute.

The Barnes-Hut kernel reached approximately

```text
Compute (SM) throughput: 83 %
```

on the development RTX 3070, while reported DRAM utilization remained comparatively low.

The result indicates that the traversal is primarily limited by computation, control flow, and latency rather than by raw DRAM bandwidth.

Nsight Compute also measured approximately

```text
10.7 active threads per 32-thread warp
```

during traversal.

The remaining divergence is expected for Barnes-Hut because different target particles eventually follow different paths through the octree even after Morton ordering.

This represents one of the principal opportunities for future optimization.

---

# 19. Benchmark results

Representative benchmarks were performed on:

```text
GPU:       NVIDIA GeForce RTX 3070 8 GB
CUDA:      12.5
Platform:  WSL2 / Linux
Precision: double
theta:     0.5
dt:        0.001
softening: 1e-6
```

Each result below is the median of three runs with 10 simulation steps.

| Particles | Average time per step |
|---:|---:|
| 100,000 | approximately **249 ms** |
| 250,000 | approximately **798 ms** |

The Morton-ordered physical particle layout was measured separately against the immediately preceding optimized version.

| Particles | Previous layout | Morton physical layout | Improvement |
|---:|---:|---:|---:|
| 100,000 | 267.96 ms/step | 249.17 ms/step | ~7.0% |
| 250,000 | 1031.84 ms/step | 797.90 ms/step | ~22.7% |

These measurements include the gather operation required to construct the Morton-ordered physical buffers. They therefore represent a complete simulation-step improvement rather than an isolated kernel measurement.

The improvement increases with particle count, making the optimized layout particularly valuable for the larger systems targeted by the project.

Benchmark timings are hardware- and dataset-dependent and should not be interpreted as universal performance values.

---

# 20. Validation

Correctness was treated as a separate requirement from performance.

The current test suite contains **111 tests** covering CPU components, CUDA kernels, complete GPU pipelines, numerical integration, input validation, and physical accuracy.

Important tested cases include:

- global bounding-box computation;
- Morton coordinate normalization and quantization;
- Morton key generation;
- radix sorting and permutation preservation;
- duplicate Morton keys;
- Morton leaf groups;
- Karras radix-tree construction;
- exact tree topology for small deterministic inputs;
- random tree construction;
- radix-tree to octree conversion;
- sparse-octree construction;
- node geometry;
- leaf mass and center of mass;
- bottom-up internal-node center of mass;
- Barnes-Hut traversal;
- exact Barnes-Hut behavior for \(\theta=0\);
- approximation of distant clusters;
- opening of asymmetric clusters;
- singular-interaction detection;
- configurable softening;
- configurable \(\theta\);
- leapfrog integration;
- two-body orbital motion;
- generated deterministic particle systems;
- solar-system ephemeris comparison;
- particle input validation;
- NaN and infinity rejection;
- negative-mass rejection;
- zero-mass particle support;
- command-line simulation behavior.

The complete suite passes after the final Morton-layout optimization.

---

# 21. CUDA memory and race validation

CUDA tests were additionally executed under NVIDIA Compute Sanitizer.

The project was checked with:

- `memcheck`;
- full leak checking;
- `racecheck`.

The final validation produced:

```text
0 memory errors
0 reported leaks
0 race hazards
```

The sanitizer runtime is significantly longer than normal test execution because CUDA kernels are instrumented. Sanitizer execution times therefore do not represent normal simulation performance.

---

# 22. Numerical accuracy

Barnes-Hut introduces a controlled force approximation through \(\theta\), whereas leapfrog introduces a time-discretization error through $\Delta t$.

These two parameters control different sources of error.

Reducing $\theta$:

- opens more octree nodes;
- approaches direct N-body force evaluation;
- increases runtime.

Reducing $\Delta t$:

- improves temporal resolution;
- reduces leapfrog integration error;
- increases the number of required simulation steps.

The test suite therefore includes both force-accuracy tests and time-integration tests.

Two-body orbit tests verify the interaction between force evaluation and leapfrog integration, while the solar-system ephemeris test compares a simulated state against an independently generated reference epoch.

---

# 23. Computational complexity

The direct N-body algorithm requires

$$
O(N^2)
$$

pair interactions per timestep.

Barnes-Hut instead builds a hierarchy of spatial cells and, for typical non-pathological distributions, reduces force evaluation toward approximately

$$
O(N\log N).
$$

The project additionally performs:

- Morton-key construction;
- radix sorting;
- radix-tree construction;
- octree conversion;
- physical-data propagation.

These operations are GPU-parallel and scale linearly or near-linearly with the number of particles for the fixed-width Morton representation.

The spatial structure must currently be rebuilt after every drift step because particle positions have changed.

---

# 24. Current limitations and possible future work

The project intentionally prioritizes correctness and a complete GPU implementation over more invasive research-level optimizations.

Several extensions remain possible.

### Warp-cooperative traversal

Despite Morton ordering, Nsight Compute still reports substantial control-flow divergence during Barnes-Hut traversal.

A future implementation could assign one spatial packet to an entire warp and use CUDA warp primitives such as

```text
__ballot_sync
__any_sync
__all_sync
```

to make cooperative tree-traversal decisions.

This could improve the low number of simultaneously active threads measured during the current independent-thread traversal.

### Improved octree memory locality

The remaining uncoalesced memory accesses are largely associated with threads visiting different octree nodes.

Reordering octree nodes according to traversal or spatial order could improve cache locality.

### Reusable tree allocations

The current implementation stages a new spatial structure before replacing the previous one.

This provides strong failure safety but requires multiple CUDA allocations during every rebuild.

Capacity-based reusable buffers could reduce allocation overhead.

### Further Barnes-Hut MAC optimization

Part of the acceptance criterion depends only on node data and could be precomputed when the octree physical data is generated.

This would trade a small amount of memory for fewer repeated floating-point operations during traversal.

### Conserved-quantity monitoring

Long simulations could additionally report:

- total energy;
- total linear momentum;
- total angular momentum;

and track their relative drift over time.

### Multi-GPU execution

The current force solver targets one CUDA GPU.

Large cluster simulations could be extended through spatial domain decomposition, MPI communication, and one GPU per MPI rank.

---

# 25. Repository structure

The main project directories are organized as follows:

```text
apps/
    simulation/          CUDA simulation executable
    viewer/              Metal visualization application
    stream_server/       frame-streaming server
    stream_client/       frame-streaming client

gpu/cuda_kernels/
    barnes_hut.cu
    globalbounding.cu
    integrator.cu
    morton.cu
    morton_leaf_groups.cu
    octree_builder.cu
    octree_geometry.cu
    octree_physics.cu
    radix_to_octree.cu
    simulation.cu
    tree_builder.cu

src/
    public interfaces and data structures
    simulation orchestration
    particle loading
    streaming implementation

tests/
    unit/
    integration/
    scripts/

data/
    synthetic-data generation
    solar-system data generation
```

The separation between topology construction, physical node data, Barnes-Hut traversal, and integration makes the individual stages independently testable.

---

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
The input must use the binary particle format documented below. The executable
advances particles in memory unless `--stream-dir` or `--cluster`
enables graphical snapshots.

## Metal viewer

The playback-capable native macOS application lives in `apps/viewer`. It can
follow a live stream, loop an archived simulation, or inspect one snapshot. It
is separate from the simulator and never opens as a side effect of running
tests.

Configure and build it from the repository root:

```bash
cmake -S . -B build/viewer -G Ninja \
  -DBUILD_TESTING=OFF \
  -DNBODY_ENABLE_CUDA=OFF \
  -DNBODY_BUILD_VIEWER=ON
cmake --build build/viewer --target nbody_viewer
```

### Replay a completed simulation

Pass the directory containing a run's `.nbsnap` files. Frames are ordered by
their sequence number and loop at the selected rate; live-stream marker files
are not required.

For example, replay the archived simulation in the Desktop presentation folder
at five frames per second:

```bash
./build/viewer/nbody_viewer \
  --replay-dir "/Users/wedoi/Desktop/presentation" \
  --replay-fps 5
```

If a directory contains snapshots from more than one run, the viewer selects
the run associated with the newest snapshot and loops only that run.

### Follow a live simulation

Use `--stream-dir` with the local output directory used by
`nbody_stream_client`:

```bash
./build/viewer/nbody_viewer --stream-dir received-frames
```

The viewer follows `.nbody-latest` while the simulation is active. When the
launcher marks the stream as ended, it automatically loops the completed run.

### Viewer flags

| Flag | Default | Description |
| --- | --- | --- |
| `--replay-dir DIR`, `--replay-dir=DIR` | none | Loops the newest archived `.nbsnap` run found in `DIR`. |
| `--stream-dir DIR`, `--stream-dir=DIR` | none | Follows completed snapshots downloaded into `DIR`, then replays the run when its stream ends. |
| `-f FILE`, `--file FILE`, `--file=FILE` | `data/test_spiral.bin` | Displays one `.nbsnap` snapshot or one planar particle file. |
| `--replay-fps FPS` | `30` | Sets completed-run playback speed. `FPS` must be positive. |
| `--poll-ms MS` | `16` | Sets how often live-stream marker files are checked. `MS` must be positive. |
| `--validate-only` | off | Loads and validates the selected file, or the newest frame in a selected directory, prints its metadata, and exits without opening a window. |
| `-h`, `--help` | — | Prints the viewer's command-line help. |

`--replay-dir`, `--stream-dir`, and `--file` are mutually exclusive. With none
of them supplied, the viewer opens `data/test_spiral.bin`. Snapshot inputs may
use either the original version-1 format or the typed version-2 format; planar
particle files are also supported.

### Viewer controls

- Drag: rotate the system.
- Two-finger scroll or arrow keys: pan; the bodies follow the gesture direction.
- Pinch or `+`/`-`: zoom.
- Click a body: follow that sampled particle across frames.
- Escape: stop following the selected body.
- `F`: fit the complete system in the window.
- `R`: reset rotation.

The title bar reports live or replay mode, the current and total simulation
steps when available, simulated time, and displayed particle count.

## Cluster browser viewer

`nbody_browser_viewer` serves an interactive WebGL 2 viewer from a cluster node
without using a cluster GPU for rendering. It watches a simulator frame
directory, displays the newest completed frame while the run is active, and
automatically loops the run after the simulator publishes its completion
marker. Particle types are rendered distinctly as stars, planets, moons,
asteroids/small bodies, or legacy unknown bodies.

### Start the viewer

Run the viewer on a node that can read the frame directory. The convenience
wrapper configures and builds the host-only executable when necessary, then
keeps the HTTP server in the foreground:

```bash
./scripts/run-browser-viewer.sh /shared/path/to/run-frames \
  --bind 127.0.0.1 \
  --port 8080
```

Keep this process running for as long as the browser should remain connected.
The printed cluster URL describes the listener, but the local browser should
normally use the forwarded `127.0.0.1` URL described below.

For a live simulation, start the simulator and viewer inside the same scheduled
job so they share the compute node and frame directory:

```bash
RUN_DIR="${SCRATCH:-$PWD}/nbody-browser-${PBS_JOBID:-manual}"
mkdir -p "$RUN_DIR"

./build/simulation/nbody \
  --input data/initial.bin \
  --time-step 0.001 \
  --steps 100000 \
  --stream-dir "$RUN_DIR" \
  --sample-rate 30 &
SIMULATION_PID=$!

./scripts/run-browser-viewer.sh "$RUN_DIR" \
  --bind 127.0.0.1 \
  --port 8080 &
VIEWER_PID=$!

printf 'Viewer compute node: %s\n' "$(hostname)"
wait "$SIMULATION_PID"
printf 'Simulation complete; viewer continues loop playback.\n'
wait "$VIEWER_PID"
```

Use `--stream-dir` without `--cluster` for retained browser playback. The raw
transfer server started by `--cluster` removes a frame after its download
client acknowledges it.

### Open an SSH tunnel

Connect the VPN first when the cluster is reachable only from its institutional
network. Do not open the cluster listener directly to the Internet.

If the viewer runs on the same login host reached by SSH, open a second terminal
on the local machine and run:

```bash
ssh -o ExitOnForwardFailure=yes \
  -N \
  -L 8080:127.0.0.1:8080 \
  <username>@<login-host>
```

The tunnel command normally prints nothing while it is working. Keep it open
and visit `http://127.0.0.1:8080` in the local browser.

If the viewer runs on a compute node and SSH access to that node is supported,
forward through the login host with `ProxyJump`:

```bash
ssh -o ExitOnForwardFailure=yes \
  -J <username>@<login-host> \
  -N \
  -L 8080:127.0.0.1:8080 \
  <username>@<compute-host>
```

`<compute-host>` is the value printed by `hostname` inside the scheduled job.
This form permits the viewer to remain bound to compute-node loopback.

If the cluster prohibits SSH sessions to compute nodes, bind the viewer to the
compute node's internal interface instead:

```bash
./scripts/run-browser-viewer.sh "$RUN_DIR" \
  --bind 0.0.0.0 \
  --port 8080
```

Then forward from the login host to that compute node:

```bash
ssh -o ExitOnForwardFailure=yes \
  -N \
  -L 8080:<compute-host>:8080 \
  <username>@<login-host>
```

This fallback exposes the viewer port to the internal cluster network, so use
it only when the cluster firewall isolates compute nodes. The viewer has no
built-in authentication.

If local port 8080 is already occupied, change only the first port in `-L`, for
example `-L 8081:127.0.0.1:8080`, and open
`http://127.0.0.1:8081`.

### Verify and troubleshoot the connection

First verify the viewer on the node where it was launched:

```bash
pgrep -af nbody_browser_viewer
ss -ltn | grep ':8080'
curl http://127.0.0.1:8080/api/health
```

Then verify the tunnel on the local machine:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
curl http://127.0.0.1:8080/api/health
```

A browser `connection refused`, `connection reset`, or SSH
`open failed: connect failed: Connection refused` message normally means the
SSH tunnel itself is present but its remote destination is wrong: the viewer is
not running, the port numbers differ, or the tunnel targets the login node
while the viewer runs on a compute node. Check the viewer process, listener,
compute hostname, and both sides of the `-L` mapping in that order.

Browser controls mirror the native viewer: drag to orbit, Shift-drag or
right-drag to shift the focus, arrow keys or horizontal/vertical two-finger
scrolling to pan, Control+scroll or Control+Plus/Minus to zoom, `R` to
fit/reset the view, and Space/timeline/speed controls for completed-run
playback.

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

The equivalent convenience wrappers configure and build their application if
needed, then launch it with the defaults above:

```bash
# On the cluster: serve the current job's persistent spool.
./scripts/run-stream-server.sh

# On the local computer: follow frames through the SSH tunnel.
./scripts/run-stream-client.sh

# Download the current backlog and exit.
./scripts/run-stream-client.sh --once
```

Useful overrides include `NBODY_STREAM_HOST`, `NBODY_STREAM_PORT`,
`NBODY_STREAM_SPOOL_DIR`, `NBODY_FRAME_OUTPUT_DIR`, and `NBODY_SKIP_BUILD=1`.
The client defaults to `received-frames` in the project directory; the server
defaults to `$SCRATCH/nbody-frames-$SLURM_JOB_ID` when those cluster variables
are available.

Each current `.nbsnap` is little-endian and independently loadable. Version 3
uses a 128-byte header containing the `NBSNAP01` magic, frame and simulation
sequence data, particle counts, payload metadata, and per-axis position bounds.
Positions are interleaved quantized `uint16` values and are followed by one
stable `uint8` particle type per sampled body. This reduces position bandwidth
to six bytes per particle while leaving the simulator's double-precision state
unchanged.

The shared frame reader, Metal viewer, and browser viewer remain compatible
with position-only version-1 float snapshots, archived typed version-2 float
snapshots with 80-byte headers, and position-only quantized version-2 snapshots
with 120-byte headers. Filenames include a unique run ID, so an older offline
backlog cannot be overwritten by a later simulation. A successful simulator
run atomically publishes `run-<id>.complete`, which tells the browser viewer to
switch from newest-frame following to loop playback.

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

## Generate Galaxy Data

The script `data/generate_galaxy.py` creates synthetic binary datasets for the galaxy viewer and the large-particle loader. Particles generated by the galaxy and two-body modes are classified as stars.

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
- then one ```uint8``` particle type for every particle

Each coordinate and velocity component is written as little-endian ```double```.
Particle types use stable values: `0` unknown, `1` star, `2` planet, `3` moon,
and `4` asteroid. The C++ loader and Metal viewer also accept legacy files
without the final type block; those particles receive the `unknown` type.

Use a writable output path such as ```data/file.bin```. A path like ```/data/file.bin``` points to a root-level directory and will usually fail with a permission error.

## Download Solar-System Data

The script `data/download_solar_system.py` builds two loader-ready datasets at user-selected TDB epochs. It combines barycentric ICRF vectors for the Sun, planet centers, and all planetary moons listed by JPL HORIZONS with the full IAU Minor Planet Center asteroid catalogue.

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

Each solar-system binary stores a type for every particle: the Sun is a star,
planet centers are planets, planetary satellites are moons, and MPC catalogue
objects are asteroids. The viewer displays stars in yellow, planets in blue,
moons in light gray, asteroids in brown, and legacy/unknown particles in white.

Downloads and individual HORIZONS responses are cached under `.cache/solar_system`, so rerunning an interrupted build resumes from the cache. Use `--refresh` to update the source files, `--offline` to require cached data, and `--overwrite` to replace existing outputs. For a quick pipeline check before processing the million-scale catalogue, add `--max-asteroids 1000`.

Major-body positions and velocities are HORIZONS geometric states. MPC asteroids are propagated independently from their heliocentric osculating elements with a two-body Kepler model, rotated from the J2000 ecliptic to ICRF, and translated using the HORIZONS barycentric Sun state. Known JPL `GM` values are converted to mass using the same `G` as the simulator. Most asteroid masses are not measured, so the script estimates them from absolute magnitude using configurable albedo and density assumptions. Moons without a published JPL `GM` are retained as massless test particles. These approximations make the files suitable for cluster-scale application and convergence testing, but the second dataset is not a high-precision HORIZONS reference trajectory for every asteroid.

The CUDA integration suite contains `SolarSystemEphemerisTest`. Its compiled
C++ fixture writer creates two compact Sun-and-planets datasets at runtime for
2026-08-26 and 2026-08-27 TDB, without Python, network access, or committed
binary data. Epoch A initializes the production simulation; after one simulated
day, all resulting 3D positions and velocities are checked against epoch B.
Per-body tolerances account for the moons deliberately omitted from this test.

## External tool

We have an external Python script that exports images and plots from simulation output.

# References

All the papers we followed and took insipation from are in the repository: 

- FMMsimple.pdf
- NBody_Description.pdf
- NumericalSolNBody.pdf
- karras2012hpg_paper.pdf
- Tree construction bonsai algo.pdf
