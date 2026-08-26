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
executable currently advances the particles in memory and reports timing and
completion information; snapshot/checkpoint output is not implemented yet.

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
