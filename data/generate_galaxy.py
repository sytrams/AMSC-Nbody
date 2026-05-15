#!/usr/bin/env python3
"""Use it like this:

  - python3 data/generate_galaxy.py --galaxy globular -n 100000 -o data/andromeda_like.bin
  - python3 data/generate_galaxy.py --galaxy spiral -n 100000 -o data/milky_way_like.bin
  - python3 data/generate_galaxy.py --galaxy spiral --shape heart -n 100000 -o data/spiral_heart.bin
  - python3 data/generate_galaxy.py --galaxy globular --shape smile -n 100000 -o data/globular_smile.bin"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic galaxy particle data in the project's binary "
            "layout: uint64 N, then x[N], y[N], z[N], vx[N], vy[N], vz[N]."
        )
    )
    parser.add_argument(
        "--galaxy",
        choices=("globular", "spiral"),
        required=True,
        help="Galaxy morphology to generate.",
    )
    parser.add_argument(
        "--shape",
        choices=("natural", "heart", "smile"),
        default="natural",
        help="Projected XY shape to apply to the galaxy.",
    )
    parser.add_argument(
        "-n",
        "--num-particles",
        type=int,
        default=100_000,
        help="Number of particles to generate.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output binary path. Defaults to data/<galaxy>_<num>.bin.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducible datasets.",
    )
    parser.add_argument(
        "--num-arms",
        type=int,
        default=4,
        help="Number of spiral arms to generate when --galaxy spiral is used.",
    )
    return parser.parse_args()


def make_output_path(galaxy: str, shape: str, num_particles: int, output: Path | None) -> Path:
    if output is not None:
        return output
    suffix = f"{galaxy}_{num_particles}" if shape == "natural" else f"{galaxy}_{shape}_{num_particles}"
    return Path(__file__).resolve().parent / f"{suffix}.bin"


def validate_particle_count(num_particles: int) -> None:
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    if num_particles > np.iinfo(np.uint32).max:
        raise ValueError("num_particles exceeds the uint32 header limit")


def validate_num_arms(num_arms: int) -> None:
    if num_arms <= 0:
        raise ValueError("num_arms must be positive")


def sample_isotropic_directions(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = rng.uniform(0.0, 2.0 * np.pi, count)
    cos_theta = rng.uniform(-1.0, 1.0, count)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta


def sample_heart_shape(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    accepted = 0

    while accepted < count:
        batch = max(4096, 4 * (count - accepted))
        x = rng.uniform(-1.3, 1.3, batch)
        y = rng.uniform(-1.2, 1.25, batch)
        heart = (x * x + y * y - 1.0) ** 3 - x * x * y * y * y
        mask = heart <= 0.0

        if np.any(mask):
            x_chunks.append(x[mask])
            y_chunks.append(y[mask])
            accepted += int(np.count_nonzero(mask))

    x_all = np.concatenate(x_chunks)[:count]
    y_all = np.concatenate(y_chunks)[:count]
    return x_all, y_all


def sample_smile_shape(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    theta = rng.uniform(np.deg2rad(205.0), np.deg2rad(335.0), count)
    radius = rng.uniform(0.78, 1.02, count)
    x = 1.18 * radius * np.cos(theta)
    y = 0.90 * radius * np.sin(theta) + 0.18
    return x, y


def apply_shape(
    rng: np.random.Generator,
    galaxy: str,
    shape: str,
    components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if shape == "natural":
        return components

    x, y, z, vx, vy, vz = components
    num_particles = x.size
    planar_radius = np.sqrt(x * x + y * y)
    planar_speed = np.sqrt(vx * vx + vy * vy)

    target_radius = np.percentile(planar_radius, 92)
    if target_radius <= 0.0:
        target_radius = 1.0

    if shape == "heart":
        shape_x, shape_y = sample_heart_shape(rng, num_particles)
        shape_scale = 0.90 * target_radius
        thickness_scale = 0.30
    else:
        shape_x, shape_y = sample_smile_shape(rng, num_particles)
        shape_scale = target_radius
        thickness_scale = 0.16

    new_x = shape_scale * shape_x
    new_y = shape_scale * shape_y
    new_z = thickness_scale * z

    radius_xy = np.sqrt(new_x * new_x + new_y * new_y) + 1.0e-9
    tangent_x = -new_y / radius_xy
    tangent_y = new_x / radius_xy
    radial_x = new_x / radius_xy
    radial_y = new_y / radius_xy

    if galaxy == "spiral":
        radial_dispersion = rng.normal(0.0, 0.10 * planar_speed + 2.0)
    else:
        radial_dispersion = rng.normal(0.0, 0.35 * planar_speed + 4.0)

    new_vx = planar_speed * tangent_x + radial_dispersion * radial_x
    new_vy = planar_speed * tangent_y + radial_dispersion * radial_y

    return new_x, new_y, new_z, new_vx, new_vy, vz.copy()


def generate_globular_galaxy(
    rng: np.random.Generator, num_particles: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale_radius = 7.5
    velocity_scale = 140.0

    # Plummer-like sphere for a dense core with a long outer tail.
    u = np.clip(rng.uniform(1.0e-12, 1.0, num_particles), 1.0e-12, 1.0 - 1.0e-12)
    radius = scale_radius / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    dir_x, dir_y, dir_z = sample_isotropic_directions(rng, num_particles)

    x = radius * dir_x
    y = radius * dir_y
    z = radius * dir_z

    sigma = velocity_scale / np.sqrt(1.0 + (radius / scale_radius) ** 2)
    vx = rng.normal(0.0, sigma)
    vy = rng.normal(0.0, sigma)
    vz = rng.normal(0.0, sigma)

    # A small coherent rotation keeps the system from being purely pressure-supported.
    radius_xy = np.sqrt(x * x + y * y) + 1.0e-9
    spin = 0.15 * velocity_scale * np.tanh(radius_xy / (2.0 * scale_radius))
    vx += -spin * y / radius_xy
    vy += spin * x / radius_xy

    return x, y, z, vx, vy, vz


def generate_spiral_galaxy(
    rng: np.random.Generator, num_particles: int, num_arms: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale_length = 8.5
    scale_height = 0.40
    arm_pitch = 1.55
    arm_width = 0.14
    circular_speed = 240.0

    bulge_fraction = 0.14
    diffuse_fraction = 0.18
    bulge_count = int(num_particles * bulge_fraction)
    diffuse_count = int(num_particles * diffuse_fraction)
    arm_count = num_particles - bulge_count - diffuse_count

    arm_radius = rng.gamma(shape=2.2, scale=scale_length, size=arm_count)
    arm_index = rng.integers(0, num_arms, size=arm_count)
    arm_phase = arm_index * (2.0 * np.pi / num_arms)
    arm_jitter = rng.normal(0.0, arm_width + 0.015 * np.sqrt(arm_radius), arm_count)
    arm_theta = arm_phase + arm_pitch * np.log1p(arm_radius / scale_length) + arm_jitter

    diffuse_radius = rng.exponential(scale=1.2 * scale_length, size=diffuse_count)
    diffuse_theta = rng.uniform(0.0, 2.0 * np.pi, diffuse_count)

    bulge_radius = np.abs(rng.normal(0.0, 0.23 * scale_length, bulge_count))
    bulge_theta = rng.uniform(0.0, 2.0 * np.pi, bulge_count)

    radius = np.concatenate((arm_radius, diffuse_radius, bulge_radius))
    theta = np.concatenate((arm_theta, diffuse_theta, bulge_theta))

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    arm_z = rng.normal(0.0, scale_height * (1.0 + 0.025 * arm_radius), arm_count)
    diffuse_z = rng.normal(0.0, 1.35 * scale_height * (1.0 + 0.03 * diffuse_radius), diffuse_count)
    bulge_z = rng.normal(0.0, 0.75 * scale_length / 3.0, bulge_count)
    z = np.concatenate((arm_z, diffuse_z, bulge_z))

    v_phi = circular_speed * (1.0 - np.exp(-radius / (0.33 * scale_length)))
    radial_drift = np.concatenate((
        rng.normal(0.0, 9.0 + 0.45 * np.sqrt(arm_radius), arm_count),
        rng.normal(0.0, 18.0, diffuse_count),
        rng.normal(0.0, 32.0, bulge_count),
    ))
    tangential_noise = np.concatenate((
        rng.normal(0.0, 10.0 + 0.35 * np.sqrt(arm_radius), arm_count),
        rng.normal(0.0, 22.0, diffuse_count),
        rng.normal(0.0, 38.0, bulge_count),
    ))
    vz = np.concatenate((
        rng.normal(0.0, 7.0, arm_count),
        rng.normal(0.0, 11.0, diffuse_count),
        rng.normal(0.0, 18.0, bulge_count),
    ))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    vx = radial_drift * cos_theta - (v_phi + tangential_noise) * sin_theta
    vy = radial_drift * sin_theta + (v_phi + tangential_noise) * cos_theta

    return x, y, z, vx, vy, vz


def write_binary_galaxy(
    output_path: Path,
    components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    x, y, z, vx, vy, vz = components
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as handle:
            handle.write(struct.pack("<Q", x.size))
            for component in (x, y, z, vx, vy, vz):
                np.asarray(component, dtype="<f8").tofile(handle)
    except OSError as exc:
        hint = ""
        if output_path.is_absolute():
            relative_hint = Path("data") / output_path.name
            hint = f" Try a writable project path such as '{relative_hint}'."
        raise RuntimeError(f"Cannot write output file '{output_path}': {exc}.{hint}") from exc


def main() -> None:
    args = parse_args()
    validate_particle_count(args.num_particles)
    validate_num_arms(args.num_arms)

    rng = np.random.default_rng(args.seed)

    if args.galaxy == "globular":
        components = generate_globular_galaxy(rng, args.num_particles)
    else:
        components = generate_spiral_galaxy(rng, args.num_particles, args.num_arms)

    components = apply_shape(rng, args.galaxy, args.shape, components)

    output_path = make_output_path(args.galaxy, args.shape, args.num_particles, args.output)
    write_binary_galaxy(output_path, components)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Created {args.galaxy} galaxy dataset with {args.shape} shape: {output_path}")
    print(f"Particles: {args.num_particles}")
    if args.galaxy == "spiral":
        print(f"Spiral arms: {args.num_arms}")
    print("Layout: uint64 N, then x[N], y[N], z[N], vx[N], vy[N], vz[N] as float64")
    print(f"Size: {file_size_mb:.2f} MiB")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
