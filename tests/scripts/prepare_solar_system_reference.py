#!/usr/bin/env python3
"""Prepare two loader-ready ephemeris datasets for the CUDA integration test."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path


PLANET_IDS = {199, 299, 399, 499, 599, 699, 799, 899, 999}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def write_horizons_cache(
    cache_dir: Path, epochs: list[float], bodies: list[dict[str, object]]
) -> None:
    tag = "-".join(f"{epoch:.6f}".replace(".", "_") for epoch in epochs)
    vector_dir = cache_dir / "horizons_vectors" / tag
    vector_dir.mkdir(parents=True, exist_ok=True)

    for body in bodies:
        states = body["states_si"]
        rows = []
        for epoch, state in zip(epochs, states, strict=True):
            values = ", ".join(f"{float(value) / 1_000.0:.17e}" for value in state)
            rows.append(f"{epoch:.15f}, pinned fixture, {values},")
        result = "\n$$SOE\n" + "\n".join(rows) + "\n$$EOE\n"
        response = {
            "signature": {
                "source": "NASA/JPL Horizons API",
                "version": "pinned-test-fixture",
            },
            "result": result,
        }
        (vector_dir / f"{int(body['spkid'])}.json").write_text(
            json.dumps(response), encoding="utf-8"
        )


def write_source_files(
    source_dir: Path, bodies: list[dict[str, object]]
) -> tuple[Path, Path, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    major_bodies = source_dir / "major_bodies.txt"
    gm_values = source_dir / "gm_Horizons.pck"
    mpcorb = source_dir / "mpcorb_empty.json.gz"

    major_bodies.write_text(
        "".join(
            f"{int(body['spkid']):9d}  {str(body['name']):<34} {'':<11}\n"
            for body in bodies
        ),
        encoding="utf-8",
    )
    gm_values.write_text(
        "".join(
            f"BODY{int(body['spkid'])}_GM = ( {float(body['gm_km3_s2']):.17e} )\n"
            for body in bodies
        ),
        encoding="utf-8",
    )
    with gzip.open(mpcorb, "wt", encoding="utf-8") as catalogue:
        catalogue.write("[]")
    return major_bodies, gm_values, mpcorb


def main() -> int:
    args = parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    epochs = [float(value) for value in fixture["epochs_jd_tdb"]]
    bodies = fixture["bodies"]

    if len(epochs) != 2 or epochs[0] == epochs[1]:
        raise ValueError("fixture must contain two distinct epochs")
    identifiers = {int(body["spkid"]) for body in bodies}
    if 10 not in identifiers or not PLANET_IDS.issubset(identifiers):
        raise ValueError("fixture must contain the Sun and all planet centers")
    if any(len(body["states_si"]) != 2 for body in bodies):
        raise ValueError("every body must have one state at each epoch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "sources"
    cache_dir = args.output_dir / "cache"
    dataset_dir = args.output_dir / "datasets"
    major_bodies, gm_values, mpcorb = write_source_files(source_dir, bodies)
    write_horizons_cache(cache_dir, epochs, bodies)

    command = [
        sys.executable,
        str(args.generator),
        "--epoch-a",
        f"{epochs[0]:.15f}",
        "--epoch-b",
        f"{epochs[1]:.15f}",
        "--output-dir",
        str(dataset_dir),
        "--cache-dir",
        str(cache_dir),
        "--mpcorb-file",
        str(mpcorb),
        "--gm-file",
        str(gm_values),
        "--major-bodies-file",
        str(major_bodies),
        "--horizons-workers",
        "1",
        "--prefix",
        "ephemeris_reference",
        "--offline",
        "--overwrite",
    ]
    subprocess.run(command, check=True)

    metadata_path = dataset_dir / "ephemeris_reference_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["particle_count"] != len(bodies):
        raise RuntimeError("generated dataset has an unexpected particle count")
    if metadata["asteroid_count"] != 0:
        raise RuntimeError("ephemeris test fixture must not contain asteroids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
