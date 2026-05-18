# How `src/particle.cppm` works

`src/particle.cppm` exports a small C++20 module named `particle`. Its purpose is to load a binary particle dataset into a `Particles` object using a structure-of-arrays layout.

This file is easiest to understand in two passes:

1. what the module defines
2. what the loader actually accepts and reads

## Module structure

The file begins with:

```cpp
module;
```

That global module fragment allows normal `#include` directives before the named module declaration:

```cpp
export module particle;
```

So this file is the implementation unit for the `particle` module.

It also exports two constants:

```cpp
export inline constexpr double G = 6.67430e-11;
export inline constexpr double K = 8.98755e9;
```

- `G` is the gravitational constant.
- `K` is the Coulomb constant.

Any translation unit that does `import particle;` can use them.

## The `Particles` class

The exported class is:

```cpp
export class Particles
```

It stores one particle system in structure-of-arrays form:

```cpp
uint32_t num_particles;
std::unique_ptr<double[]> mass;
std::unique_ptr<double[]> x, y, z;
std::unique_ptr<double[]> vx, vy, vz;
```

So instead of storing one struct per particle, it stores seven separate contiguous arrays:

- `mass[i]`
- `x[i]`, `y[i]`, `z[i]`
- `vx[i]`, `vy[i]`, `vz[i]`

This layout is common in numerical code because each component is contiguous in memory.

## The private header helper

Inside an anonymous namespace, the file defines:

```cpp
struct ParticleFileHeader {
    uint32_t num_particles;
    std::streamoff header_bytes;
};
```

and the helper:

```cpp
ParticleFileHeader detect_particle_file_header(std::ifstream& inFile)
```

This function does not read the particle arrays themselves. It only tries to determine:

- how many particles are in the file
- whether the header is 4 bytes or 8 bytes long

## What header formats it accepts

The helper supports two count headers:

- `uint64_t`
- `uint32_t`

The logic is:

1. seek to the end and get the total file size
2. rewind to the beginning
3. try reading the first 8 bytes as `uint64_t`
4. if the file size matches the expected layout, accept that format
5. otherwise rewind again and try `uint32_t`
6. if neither matches, throw an exception

The key check is the file-size validation.

## The important mismatch

This is the most important part of the file.

The validator computes:

```cpp
constexpr std::streamoff bytes_per_particle =
    static_cast<std::streamoff>(6 * sizeof(double));
```

That means `detect_particle_file_header(...)` assumes the binary file contains exactly six `double` arrays after the header.

So the accepted on-disk layout is:

```text
[count][a0...aN-1][b0...bN-1][c0...][d0...][e0...][f0...]
```

with six blocks total.

This matches the current README and the current test generator, which both describe/write:

- `x`
- `y`
- `z`
- `vx`
- `vy`
- `vz`

after the header.

## What the constructor actually reads

The constructor for `Particles` does this after validation:

1. allocate seven arrays
2. seek to the first byte after the header
3. read seven blocks of `double`

The reads are:

```cpp
inFile.read(reinterpret_cast<char*>(mass.get()), bytes_to_read);
inFile.read(reinterpret_cast<char*>(x.get()), bytes_to_read);
inFile.read(reinterpret_cast<char*>(y.get()), bytes_to_read);
inFile.read(reinterpret_cast<char*>(z.get()), bytes_to_read);
inFile.read(reinterpret_cast<char*>(vx.get()), bytes_to_read);
inFile.read(reinterpret_cast<char*>(vy.get()), bytes_to_read);
inFile.read(reinterpret_cast<char*>(vz.get()), bytes_to_read);
```

So the constructor expects this layout:

```text
[count][mass][x][y][z][vx][vy][vz]
```

which is seven `double` arrays, not six.

## What that means in practice

As written, the module is internally inconsistent.

### Case 1: a six-array file

This is the format currently documented elsewhere in the repo:

```text
[count][x][y][z][vx][vy][vz]
```

For such a file:

- header detection succeeds, because the file size matches `6 * sizeof(double)` per particle
- the constructor then reads the first block into `mass`, the second into `x`, and so on
- the final read into `vz` reaches EOF and fails

So the constructor throws:

```cpp
Failed to read all expected particle data. File may be truncated.
```

### Case 2: a seven-array file

If a file is written as:

```text
[count][mass][x][y][z][vx][vy][vz]
```

then the constructor would conceptually read it correctly, but the header detector rejects it first because its size is larger than the six-array formula.

So it throws:

```cpp
Unsupported particle file layout. Expected planar doubles with a 32-bit or 64-bit particle count header.
```

## Bottom line

`src/particle.cppm` is clearly intended to be a binary loader for a structure-of-arrays particle dataset, and the public class layout suggests the intended in-memory representation is:

```text
mass, x, y, z, vx, vy, vz
```

But the implementation currently mixes two different file formats:

- the validator still assumes the old six-array format
- the constructor already assumes a newer seven-array format with `mass`

So the correct way to read the file today is:

- the helper `detect_particle_file_header(...)` validates using the old six-array size rule
- the constructor then tries to load a seven-array payload
- those two assumptions conflict

That is why `particle.cppm` is best understood as a loader in transition from a 6-field file format to a 7-field file format.

## Short walkthrough of constructor flow

Ignoring the mismatch for a moment, the constructor logic is straightforward:

1. verify that the input stream is open and good
2. call `detect_particle_file_header(...)`
3. store `num_particles`
4. reject zero particles
5. allocate dynamic arrays
6. seek past the header
7. read one contiguous block per field
8. throw if any read fails

The raw reads use `reinterpret_cast<char*>` because `std::ifstream::read` works with byte buffers, while the destination arrays are `double*`.
