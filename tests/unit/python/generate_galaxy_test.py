#!/usr/bin/env python3
import contextlib
import io
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "data"))

import generate_galaxy  # noqa: E402


class TwoBodyGenerationTest(unittest.TestCase):
    def test_preserves_existing_random_generator_defaults(self):
        args = generate_galaxy.parse_args(["--galaxy", "spiral"])

        self.assertEqual(args.num_particles, 100_000)
        self.assertEqual(args.mass, 1.0e30)

    def test_generates_equal_mass_circular_orbit(self):
        mass_value = 1.0e10
        separation = 1.0
        expected_velocity = math.sqrt(
            generate_galaxy.GRAVITATIONAL_CONSTANT
            * mass_value
            / (2.0 * separation)
        )

        data = generate_galaxy.generate_two_body_data(mass_value, separation)
        mass, x, y, z, vx, vy, vz = data

        np.testing.assert_array_equal(mass, [mass_value, mass_value])
        np.testing.assert_array_equal(x, [-0.5, 0.5])
        np.testing.assert_array_equal(y, [0.0, 0.0])
        np.testing.assert_array_equal(z, [0.0, 0.0])
        np.testing.assert_array_equal(vx, [0.0, 0.0])
        np.testing.assert_allclose(vy, [-expected_velocity, expected_velocity])
        np.testing.assert_array_equal(vz, [0.0, 0.0])

        self.assertAlmostEqual(float(np.sum(mass * x)), 0.0)
        self.assertAlmostEqual(float(np.sum(mass * vy)), 0.0)

    def test_writes_particle_loader_binary_layout(self):
        args = generate_galaxy.parse_args(
            ["--galaxy", "two-body", "--mass", "1e10", "--separation", "2"]
        )
        data = generate_galaxy.generate_data(args)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "two-body.bin"
            generate_galaxy.write_particle_file(path, data)
            raw = path.read_bytes()

        self.assertEqual(len(raw), 8 + 2 * 7 * 8)
        self.assertEqual(struct.unpack_from("<Q", raw)[0], 2)
        stored_values = struct.unpack_from("<14d", raw, 8)
        expected_values = tuple(
            float(value) for component in data for value in component
        )
        self.assertEqual(stored_values, expected_values)

    def test_rejects_particle_count_other_than_two(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                generate_galaxy.parse_args(
                    ["--galaxy", "two-body", "--num-particles", "3"]
                )


if __name__ == "__main__":
    unittest.main()
