#!/usr/bin/env python3
import argparse
import math
import struct
from pathlib import Path

import numpy as np


GRAVITATIONAL_CONSTANT = 6.67430e-11
DEFAULT_PARTICLE_COUNT = 100_000
DEFAULT_GALAXY_MASS = 1.0e30
DEFAULT_TWO_BODY_MASS = 1.0e10


def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--galaxy", choices=("globular", "spiral", "two-body"), required=True
    )
    parser.add_argument(
        "--shape", choices=("natural", "heart", "smile"), default="natural"
    )
    parser.add_argument("-n", "--num-particles", type=positive_integer, default=None)
    parser.add_argument("--mass", type=positive_float, default=None)
    parser.add_argument(
        "--separation",
        type=positive_float,
        default=1.0,
        help="center-to-center separation for --galaxy two-body (default: 1.0)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-arms", type=positive_integer, default=4)
    args = parser.parse_args(argv)

    if args.galaxy == "two-body":
        if args.num_particles not in (None, 2):
            parser.error("--galaxy two-body requires --num-particles 2")
        if args.shape != "natural":
            parser.error("--shape is not applicable to --galaxy two-body")
        args.num_particles = 2
        if args.mass is None:
            args.mass = DEFAULT_TWO_BODY_MASS
    else:
        if args.num_particles is None:
            args.num_particles = DEFAULT_PARTICLE_COUNT
        if args.mass is None:
            args.mass = DEFAULT_GALAXY_MASS

    return args


def make_output_path(galaxy, shape, num_particles, output):
    if output is not None:
        return output
    suffix = (
        f"{galaxy}_{num_particles}"
        if shape == "natural"
        else f"{galaxy}_{shape}_{num_particles}"
    )
    return Path(__file__).resolve().parent / f"{suffix}.bin"


def generate_two_body_data(mass_value, separation):
    orbital_velocity = math.sqrt(
        GRAVITATIONAL_CONSTANT * mass_value / (2.0 * separation)
    )

    mass = np.full(2, mass_value, dtype="<f8")
    x = np.array([-0.5 * separation, 0.5 * separation], dtype="<f8")
    y = np.zeros(2, dtype="<f8")
    z = np.zeros(2, dtype="<f8")
    vx = np.zeros(2, dtype="<f8")
    vy = np.array([-orbital_velocity, orbital_velocity], dtype="<f8")
    vz = np.zeros(2, dtype="<f8")
    return mass, x, y, z, vx, vy, vz


def generate_random_data(args):
    rng = np.random.default_rng(args.seed)
    n = args.num_particles
    mass = np.full(n, args.mass, dtype="<f8")
    x = rng.normal(0, 1, n).astype("<f8")
    y = rng.normal(0, 1, n).astype("<f8")
    z = rng.normal(0, 1, n).astype("<f8")
    vx = rng.normal(0, 0.01, n).astype("<f8")
    vy = rng.normal(0, 0.01, n).astype("<f8")
    vz = rng.normal(0, 0.01, n).astype("<f8")
    return mass, x, y, z, vx, vy, vz


def generate_data(args):
    if args.galaxy == "two-body":
        return generate_two_body_data(args.mass, args.separation)
    return generate_random_data(args)


def write_particle_file(path, data):
    particle_count = len(data[0])
    if any(len(component) != particle_count for component in data):
        raise ValueError("all particle components must have the same length")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", particle_count))
        for component in data:
            component.astype("<f8", copy=False).tofile(output)


def two_body_orbital_period(mass_value, separation):
    angular_velocity = math.sqrt(
        2.0 * GRAVITATIONAL_CONSTANT * mass_value / separation**3
    )
    return 2.0 * math.pi / angular_velocity


def main():
    args = parse_args()
    data = generate_data(args)
    path = make_output_path(
        args.galaxy, args.shape, args.num_particles, args.output
    )
    write_particle_file(path, data)
    print(f"Created {path}")

    if args.galaxy == "two-body":
        velocity = abs(data[5][0])
        period = two_body_orbital_period(args.mass, args.separation)
        print(
            "Two-body circular orbit: "
            f"mass={args.mass:.17g}, separation={args.separation:.17g}, "
            f"speed={velocity:.17g}, period={period:.17g}"
        )
        print(f"Suggested timestep for 128 steps/orbit: {period / 128.0:.17g}")


if __name__ == "__main__":
    main()
