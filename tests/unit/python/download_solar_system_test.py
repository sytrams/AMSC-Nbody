#!/usr/bin/env python3
import gzip
import io
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "data"))

import download_solar_system as solar  # noqa: E402


def major_body_line(spkid, name="", designation=""):
    return f"{spkid:9d}  {name:<34} {designation:<11}"


class SourceParsingTest(unittest.TestCase):
    def test_streams_top_level_json_array(self):
        source = io.StringIO('[{"a": 1, "nested": [2, 3]}, {"name": "Ceres"}]')
        values = list(solar.iter_json_array(source, chunk_size=7))
        self.assertEqual(values, [{"a": 1, "nested": [2, 3]}, {"name": "Ceres"}])

    def test_selects_sun_planet_centers_and_planetary_moons(self):
        lines = [major_body_line(10, "Sun")]
        lines.extend(
            major_body_line(spkid, f"Planet {spkid}")
            for spkid in sorted(solar.PLANET_IDS)
        )
        lines.extend(
            (
                major_body_line(301, "Moon"),
                major_body_line(501, "Io"),
                major_body_line(55501, "", "S2003_J2"),
                major_body_line(31, "SEMB-L1"),
                major_body_line(-61, "Juno spacecraft"),
                major_body_line(120050000, "Weywot"),
            )
        )

        bodies = solar.parse_major_body_index("\n".join(lines))
        identifiers = [body.spkid for body in bodies]

        self.assertIn(10, identifiers)
        self.assertIn(301, identifiers)
        self.assertIn(501, identifiers)
        self.assertIn(55501, identifiers)
        self.assertNotIn(31, identifiers)
        self.assertNotIn(-61, identifiers)
        self.assertNotIn(120050000, identifiers)
        provisional = next(body for body in bodies if body.spkid == 55501)
        self.assertEqual(provisional.name, "S2003_J2")

    def test_parses_gravitational_parameters(self):
        values = solar.parse_gm_pck(
            "BODY10_GM = ( 1.3271244004127942E+11 )\n"
            "BODY301_GM=(4.9028001184575496E+03)\n"
        )
        self.assertEqual(values[10], 1.3271244004127942e11)
        self.assertEqual(values[301], 4902.80011845755)

    def test_parses_two_horizons_vectors_and_converts_to_si(self):
        result = """header
$$SOE
2461278.5, A.D. date, 1, 2, 3, 4, 5, 6,
2461309.5, A.D. date, 7, 8, 9, 10, 11, 12,
$$EOE
footer
"""
        vectors = solar.parse_horizons_vectors(
            json.dumps({"result": result}), [2461278.5, 2461309.5]
        )
        np.testing.assert_array_equal(
            vectors,
            np.asarray(
                [[1000, 2000, 3000, 4000, 5000, 6000],
                 [7000, 8000, 9000, 10000, 11000, 12000]],
                dtype=np.float64,
            ),
        )


class OrbitConversionTest(unittest.TestCase):
    def setUp(self):
        self.gm_values = {10: 1.3271244004127942e11, 2_000_001: 62.6290536056}
        solar_mu = self.gm_values[10] * 1.0e9
        self.mean_motion_deg_day = (
            math.degrees(math.sqrt(solar_mu / solar.AU_METERS**3))
            * solar.SECONDS_PER_DAY
        )
        self.record = {
            "Number": "(1)",
            "Name": "Test asteroid",
            "Principal_desig": "TEST",
            "H": 3.34,
            "Epoch": 2451545.0,
            "M": 0.0,
            "Peri": 0.0,
            "Node": 0.0,
            "i": 0.0,
            "e": 0.0,
            "n": self.mean_motion_deg_day,
            "a": 1.0,
        }

    def make_batch(self):
        return solar.build_asteroid_batch(
            [self.record],
            self.gm_values,
            albedo=0.14,
            density=2000.0,
            unknown_mass=0.0,
        )

    def test_uses_known_jpl_gm_for_mass(self):
        batch = self.make_batch()
        expected = self.gm_values[2_000_001] * 1.0e9 / solar.GRAVITATIONAL_CONSTANT
        self.assertAlmostEqual(batch.mass[0], expected)
        self.assertEqual(batch.mass_sources, ["JPL GM"])

    def test_propagates_circular_orbit_and_rotates_to_icrf(self):
        batch = self.make_batch()
        zero_sun_state = np.zeros(6)
        initial = solar.propagate_asteroid_batch(
            batch, self.record["Epoch"], zero_sun_state
        )[0]
        orbital_speed = math.sqrt(
            self.gm_values[10] * 1.0e9 / solar.AU_METERS
        )
        np.testing.assert_allclose(initial[:3], [solar.AU_METERS, 0.0, 0.0], rtol=1e-14)
        np.testing.assert_allclose(
            initial[3:],
            [
                0.0,
                orbital_speed * math.cos(solar.J2000_OBLIQUITY),
                orbital_speed * math.sin(solar.J2000_OBLIQUITY),
            ],
            rtol=1e-13,
            atol=1e-11,
        )

        period_days = 360.0 / self.mean_motion_deg_day
        quarter = solar.propagate_asteroid_batch(
            batch, self.record["Epoch"] + period_days / 4.0, zero_sun_state
        )[0]
        np.testing.assert_allclose(
            quarter[:3],
            [
                0.0,
                solar.AU_METERS * math.cos(solar.J2000_OBLIQUITY),
                solar.AU_METERS * math.sin(solar.J2000_OBLIQUITY),
            ],
            rtol=1e-12,
            atol=1.0,
        )


class ParticleWriterTest(unittest.TestCase):
    def test_writes_loader_compatible_planar_layout(self):
        mass = np.asarray([2.0, 3.0])
        state = np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "particles.bin"
            with solar.ParticleDatasetWriter(path) as writer:
                writer.append(mass, state)
                writer.finalize()
            raw = path.read_bytes()

        self.assertEqual(struct.unpack_from("<Q", raw)[0], 2)
        values = struct.unpack_from("<14d", raw, 8)
        self.assertEqual(
            values,
            (2.0, 3.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0,
             4.0, 10.0, 5.0, 11.0, 6.0, 12.0),
        )


if __name__ == "__main__":
    unittest.main()
