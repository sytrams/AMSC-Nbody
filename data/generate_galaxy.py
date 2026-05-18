#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxy", choices=("globular", "spiral"), required=True)
    parser.add_argument("--shape", choices=("natural", "heart", "smile"), default="natural")
    parser.add_argument("-n", "--num-particles", type=int, default=100_000)
    parser.add_argument("--mass", type=float, default=1.0e30)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-arms", type=int, default=4)
    return parser.parse_args()

def make_output_path(galaxy, shape, num_particles, output):
    if output is not None: return output
    suffix = f"{galaxy}_{num_particles}" if shape == "natural" else f"{galaxy}_{shape}_{num_particles}"
    return Path(__file__).resolve().parent / f"{suffix}.bin"

def generate_data(args):
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

def main():
    args = parse_args()
    data = generate_data(args)
    path = make_output_path(args.galaxy, args.shape, args.num_particles, args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", args.num_particles))
        for arr in data:
            arr.tofile(f)
    print(f"Created {path}")

if __name__ == '__main__':
    main()
