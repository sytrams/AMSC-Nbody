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

## particle.hpp
This header file contains the principal implementation of the Verlet Integration that calculates and updates the values of the positions and velocities of the 2 bodies interacting with each other. It also defines the Particle class template and the fundamental particle we use to simulate the dynamics of a physical system.

## system.hpp
This header file defines the gravitation_system class template, designed to simulate the dynamics of a system of N bodies (N-body problem) interacting with each other through the gravitational force, using the Verlet Integration algorithm and OpenMP parallelization.

## vector.hpp
This header file defines the Vector class template, an elementary linear algebra library for representing and manipulating fixed-dimensional (DIM) vectors.

## main.cpp
The main file manages the execution flow of the N-body simulation. Its purpose is to initialize the system with the correct spatial dimension, run the simulation over the specified time interval, and store the results. The positions of all particles (bodies) are saved every 20 iterations in an output file, which can then be used to visualize their trajectories.


## How to use it
In order to compile the program you have to run the make command inside the project root:
```
make
```

Once the program is compiled, you can move into the build directory and run the executable file
```
cd build
./nbody
```

The program takes in some input arguments, which are visible with the command
```
./nbody -h
```
```
./nbody --help
```

Alternatively, you can compile and run the program by only using the command
```
make run args = "[options]"
```
The default option to execute with simple ```make run``` command is ```-f ../src/solar_system.txt``` that give to the simulaton the initilizarion for our Solar System.

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



