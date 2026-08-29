#!/usr/bin/env python3
"""Build two AMSC-Nbody particle files from JPL and MPC data.

Major-body states come from JPL HORIZONS in barycentric ICRF coordinates.
Asteroid osculating elements come from the IAU Minor Planet Center's bulk
MPCORB catalogue and are propagated with an unperturbed two-body model.  The
catalogue is processed as a stream so a million-object run has bounded memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, TextIO

import numpy as np

from particle_types import ParticleType, VALID_PARTICLE_TYPES


HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_GM_URL = "https://ssd.jpl.nasa.gov/ftp/xfr/gm_Horizons.pck"
MPCORB_URL = "https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz"

GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2, matches particle.hpp
AU_METERS = 149_597_870_700.0
SECONDS_PER_DAY = 86_400.0
UNIX_EPOCH_JD = 2_440_587.5
J2000_OBLIQUITY = math.radians(23.439291111)
COMPONENT_NAMES = ("mass", "x", "y", "z", "vx", "vy", "vz")
PLANET_IDS = frozenset({199, 299, 399, 499, 599, 699, 799, 899, 999})
USER_AGENT = "AMSC-Nbody solar-system dataset builder/1.0"


@dataclass(frozen=True)
class Epoch:
    text: str
    jd_tdb: float


@dataclass(frozen=True)
class MajorBody:
    spkid: int
    name: str
    category: str


@dataclass
class AsteroidBatch:
    records: list[Mapping[str, object]]
    eccentricity: np.ndarray
    semimajor_axis_au: np.ndarray
    inclination: np.ndarray
    ascending_node: np.ndarray
    argument_of_perihelion: np.ndarray
    mean_anomaly: np.ndarray
    mean_motion: np.ndarray
    orbit_epoch_jd: np.ndarray
    mass: np.ndarray
    mass_sources: list[str]


class DownloadError(RuntimeError):
    pass


class ParticleDatasetWriter:
    """Stream planar particle components and atomically assemble one file."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=f".{output_path.name}.", dir=output_path.parent
        )
        temporary_root = Path(self._temporary_directory.name)
        self._paths = {
            name: temporary_root / f"{name}.f64" for name in COMPONENT_NAMES
        }
        self._paths["type"] = temporary_root / "type.u8"
        self._handles = {
            name: path.open("wb") for name, path in self._paths.items()
        }
        self.count = 0
        self._finalized = False

    def append(
        self, mass: np.ndarray, state: np.ndarray, particle_type: np.ndarray
    ) -> None:
        mass = np.asarray(mass, dtype="<f8")
        state = np.asarray(state, dtype="<f8")
        particle_type = np.asarray(particle_type, dtype="<u1")
        if state.shape != (mass.size, 6):
            raise ValueError("state must have shape (particle_count, 6)")
        if particle_type.shape != (mass.size,):
            raise ValueError("particle_type must have shape (particle_count,)")
        valid_type_values = tuple(int(value) for value in VALID_PARTICLE_TYPES)
        if not np.all(np.isin(particle_type, valid_type_values)):
            raise ValueError("particle_type contains an unsupported value")
        if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(state)):
            raise ValueError("particle data must be finite")

        mass.tofile(self._handles["mass"])
        for index, name in enumerate(COMPONENT_NAMES[1:]):
            state[:, index].astype("<f8", copy=False).tofile(self._handles[name])
        particle_type.tofile(self._handles["type"])
        self.count += mass.size

    def finalize(self) -> None:
        if self._finalized:
            return
        for handle in self._handles.values():
            handle.close()

        partial_path = self.output_path.with_name(self.output_path.name + ".part")
        try:
            with partial_path.open("wb") as output:
                output.write(struct.pack("<Q", self.count))
                for name in COMPONENT_NAMES:
                    with self._paths[name].open("rb") as component:
                        shutil.copyfileobj(component, output, length=8 * 1024 * 1024)
                with self._paths["type"].open("rb") as component:
                    shutil.copyfileobj(component, output, length=8 * 1024 * 1024)
            os.replace(partial_path, self.output_path)
            self._finalized = True
        finally:
            partial_path.unlink(missing_ok=True)

    def close(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.close()
        self._temporary_directory.cleanup()

    def __enter__(self) -> "ParticleDatasetWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = nonnegative_float(value)
    if parsed == 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_epoch(value: str) -> Epoch:
    text = value.strip()
    try:
        numeric = float(text)
    except ValueError:
        numeric = math.nan
    if math.isfinite(numeric):
        if numeric <= 0.0:
            raise argparse.ArgumentTypeError("Julian date must be positive")
        return Epoch(text=text, jd_tdb=numeric)

    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        instant = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a Julian date or ISO date such as 2026-08-26T00:00:00"
        ) from error
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    else:
        instant = instant.astimezone(timezone.utc)
    jd = instant.timestamp() / SECONDS_PER_DAY + UNIX_EPOCH_JD
    return Epoch(text=text, jd_tdb=jd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Sun/planet/moon states and the full MPC asteroid catalogue, "
            "then write two AMSC-Nbody binary datasets. Epochs are interpreted as TDB."
        )
    )
    parser.add_argument("--epoch-a", required=True, type=parse_epoch)
    parser.add_argument("--epoch-b", required=True, type=parse_epoch)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/solar_system")
    )
    parser.add_argument("--prefix", default="solar_system")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/solar_system")
    )
    parser.add_argument(
        "--mpcorb-file",
        type=Path,
        help="use an existing mpcorb_extended.json.gz instead of downloading it",
    )
    parser.add_argument(
        "--gm-file",
        type=Path,
        help="use an existing gm_Horizons.pck instead of downloading it",
    )
    parser.add_argument(
        "--major-bodies-file",
        type=Path,
        help="use a cached HORIZONS COMMAND=MB text response",
    )
    parser.add_argument("--batch-size", type=positive_integer, default=50_000)
    parser.add_argument(
        "--max-asteroids",
        type=positive_integer,
        help="limit accepted asteroids for a quick local test",
    )
    parser.add_argument(
        "--asteroid-albedo", type=positive_float, default=0.14
    )
    parser.add_argument(
        "--asteroid-density",
        type=positive_float,
        default=2_000.0,
        help="assumed density in kg/m^3 when mass is estimated from H",
    )
    parser.add_argument(
        "--unknown-asteroid-mass",
        type=nonnegative_float,
        default=0.0,
        help="mass used when an asteroid has neither a known GM nor H",
    )
    parser.add_argument(
        "--horizons-workers", type=positive_integer, default=4
    )
    parser.add_argument("--http-timeout", type=positive_float, default=120.0)
    parser.add_argument("--http-retries", type=positive_integer, default=5)
    parser.add_argument(
        "--strict-major-bodies",
        action="store_true",
        help="fail if any listed planetary moon lacks coverage at either epoch",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="make no network requests; require all source/API files in cache",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="redownload cached source/API files"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output files"
    )
    args = parser.parse_args(argv)

    if args.epoch_a.jd_tdb == args.epoch_b.jd_tdb:
        parser.error("--epoch-a and --epoch-b must be different")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.prefix):
        parser.error("--prefix may contain only letters, digits, '.', '_' and '-'")
    return args


def _request_with_retries(
    url: str, timeout: float, retries: int
) -> urllib.response.addinfourl:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
        if attempt + 1 < retries:
            time.sleep(min(2.0**attempt, 16.0))
    raise DownloadError(f"request failed after {retries} attempts: {url}") from last_error


def fetch_to_cache(
    url: str,
    path: Path,
    *,
    timeout: float,
    retries: int,
    offline: bool,
    refresh: bool,
) -> Path:
    if path.exists() and not refresh:
        return path
    if offline:
        raise DownloadError(f"offline mode requires cached file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".part")
    partial_path.unlink(missing_ok=True)
    try:
        with _request_with_retries(url, timeout, retries) as response:
            with partial_path.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(partial_path, path)
    finally:
        partial_path.unlink(missing_ok=True)
    return path


def fetch_api_text(
    parameters: Mapping[str, object],
    cache_path: Path,
    *,
    timeout: float,
    retries: int,
    offline: bool,
    refresh: bool,
) -> str:
    query = urllib.parse.urlencode(parameters)
    fetch_to_cache(
        f"{HORIZONS_API_URL}?{query}",
        cache_path,
        timeout=timeout,
        retries=retries,
        offline=offline,
        refresh=refresh,
    )
    return cache_path.read_text(encoding="utf-8")


def is_planetary_moon(spkid: int) -> bool:
    if 300 <= spkid <= 999:
        parent = spkid // 100
        return 3 <= parent <= 9 and spkid % 100 != 99
    if 10_000 <= spkid <= 99_999:
        return 3 <= spkid // 10_000 <= 9
    return False


def parse_major_body_index(text: str) -> list[MajorBody]:
    bodies: list[MajorBody] = []
    seen: set[int] = set()
    for line in text.splitlines():
        identifier = line[:9].strip()
        if not identifier or not identifier.lstrip("-").isdigit():
            continue
        spkid = int(identifier)
        if spkid == 10:
            category = "star"
        elif spkid in PLANET_IDS:
            category = "planet"
        elif is_planetary_moon(spkid):
            category = "moon"
        else:
            continue
        if spkid in seen:
            continue
        seen.add(spkid)
        name = line[11:45].strip() or line[46:58].strip() or f"SPK {spkid}"
        bodies.append(MajorBody(spkid=spkid, name=name, category=category))

    if 10 not in seen or not PLANET_IDS.issubset(seen):
        raise ValueError("HORIZONS major-body list is missing the Sun or a planet")
    category_order = {"star": 0, "planet": 1, "moon": 2}
    bodies.sort(key=lambda body: (category_order[body.category], body.spkid))
    return bodies


def parse_gm_pck(text: str) -> dict[int, float]:
    pattern = re.compile(
        r"BODY(\d+)_GM\s*=\s*\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
        re.MULTILINE,
    )
    values = {int(match.group(1)): float(match.group(2)) for match in pattern.finditer(text)}
    if 10 not in values:
        raise ValueError("gm_Horizons.pck does not define BODY10_GM")
    return values


def parse_horizons_vectors(payload_text: str, expected_jds: Sequence[float]) -> np.ndarray:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("HORIZONS response is not valid JSON") from error
    result = payload.get("result", "")
    if not isinstance(result, str) or "$$SOE" not in result or "$$EOE" not in result:
        message = payload.get("error") or str(result).strip() or "missing vector table"
        raise ValueError(f"HORIZONS did not return vectors: {message}")

    table = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    parsed: list[tuple[float, np.ndarray]] = []
    for row in csv.reader(table.splitlines()):
        if len(row) < 8 or not row[0].strip():
            continue
        try:
            jd = float(row[0])
            vector = np.asarray([float(value) for value in row[2:8]], dtype=np.float64)
        except ValueError as error:
            raise ValueError(f"invalid HORIZONS vector row: {row}") from error
        parsed.append((jd, vector))

    output = np.empty((len(expected_jds), 6), dtype=np.float64)
    for index, expected in enumerate(expected_jds):
        matches = [vector for jd, vector in parsed if abs(jd - expected) < 1.0e-8]
        if len(matches) != 1:
            raise ValueError(f"expected one HORIZONS vector at JD {expected:.12f}")
        output[index] = matches[0]
    output[:, :3] *= 1_000.0  # km -> m
    output[:, 3:] *= 1_000.0  # km/s -> m/s
    return output


def load_major_body_states(
    bodies: Sequence[MajorBody], epochs: Sequence[Epoch], args: argparse.Namespace
) -> tuple[list[MajorBody], dict[int, np.ndarray], list[dict[str, object]]]:
    expected_jds = [epoch.jd_tdb for epoch in epochs]
    tlist = " ".join(f"'{jd:.12f}'" for jd in expected_jds)
    tag = "-".join(f"{jd:.6f}".replace(".", "_") for jd in expected_jds)
    cache_root = args.cache_dir / "horizons_vectors" / tag

    def load_one(body: MajorBody) -> np.ndarray:
        parameters = {
            "format": "json",
            "COMMAND": str(body.spkid),
            "CENTER": "500@0",
            "MAKE_EPHEM": "YES",
            "OBJ_DATA": "NO",
            "EPHEM_TYPE": "VECTORS",
            "TLIST": tlist,
            "TLIST_TYPE": "JD",
            "TIME_TYPE": "TDB",
            "REF_PLANE": "FRAME",
            "REF_SYSTEM": "ICRF",
            "VEC_CORR": "NONE",
            "VEC_TABLE": "2",
            "OUT_UNITS": "KM-S",
            "CSV_FORMAT": "YES",
            "VEC_LABELS": "YES",
        }
        text = fetch_api_text(
            parameters,
            cache_root / f"{body.spkid}.json",
            timeout=args.http_timeout,
            retries=args.http_retries,
            offline=args.offline,
            refresh=args.refresh,
        )
        return parse_horizons_vectors(text, expected_jds)

    states: dict[int, np.ndarray] = {}
    failures: list[dict[str, object]] = []
    print(f"Requesting two-epoch HORIZONS vectors for {len(bodies)} major bodies...")
    with ThreadPoolExecutor(max_workers=args.horizons_workers) as executor:
        futures = {executor.submit(load_one, body): body for body in bodies}
        completed = 0
        for future in as_completed(futures):
            body = futures[future]
            try:
                states[body.spkid] = future.result()
            except Exception as error:
                failures.append(
                    {"spkid": body.spkid, "name": body.name, "error": str(error)}
                )
            completed += 1
            if completed % 25 == 0 or completed == len(bodies):
                print(f"  HORIZONS: {completed}/{len(bodies)} requests complete")

    required_failures = [
        failure
        for failure in failures
        if failure["spkid"] == 10 or failure["spkid"] in PLANET_IDS
    ]
    if required_failures:
        raise DownloadError(f"required HORIZONS body failed: {required_failures[0]}")
    if failures and args.strict_major_bodies:
        raise DownloadError(f"HORIZONS moon failed: {failures[0]}")
    if failures:
        print(
            f"Warning: skipping {len(failures)} moons without coverage at both epochs.",
            file=sys.stderr,
        )
    included = [body for body in bodies if body.spkid in states]
    return included, states, failures


def iter_json_array(stream: TextIO, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    """Incrementally decode one top-level JSON array using the standard library."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False
    while not finished:
        chunk = stream.read(chunk_size)
        if chunk:
            buffer += chunk
        elif position >= len(buffer):
            break

        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    break
                if buffer[position] != "[":
                    raise ValueError("MPCORB JSON must contain a top-level array")
                started = True
                position += 1
                continue
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position >= len(buffer):
                break
            if buffer[position] == "]":
                position += 1
                finished = True
                break
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if not chunk:
                    raise ValueError("truncated MPCORB JSON array")
                break
            yield value
            position = end

        if position:
            buffer = buffer[position:]
            position = 0
        if not chunk and not finished:
            raise ValueError("truncated MPCORB JSON array")
    if not started or not finished:
        raise ValueError("incomplete MPCORB JSON array")


def _finite_float(record: Mapping[str, object], key: str) -> float | None:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def valid_asteroid_record(record: Mapping[str, object]) -> bool:
    values = [_finite_float(record, key) for key in ("e", "a", "i", "Node", "Peri", "M", "Epoch")]
    if any(value is None for value in values):
        return False
    eccentricity, semimajor_axis = values[0], values[1]
    if eccentricity is None or semimajor_axis is None or eccentricity < 0.0:
        return False
    if abs(eccentricity - 1.0) < 1.0e-10:
        return False
    return (eccentricity < 1.0 and semimajor_axis > 0.0) or (
        eccentricity > 1.0 and semimajor_axis < 0.0
    )


def asteroid_number(record: Mapping[str, object]) -> int | None:
    number = record.get("Number")
    if number is None:
        return None
    match = re.fullmatch(r"\((\d+)\)", str(number).strip())
    return int(match.group(1)) if match else None


def known_asteroid_gm(number: int | None, gm_values: Mapping[int, float]) -> float | None:
    if number is None:
        return None
    for spkid in (2_000_000 + number, 20_000_000 + number):
        if spkid in gm_values:
            return gm_values[spkid]
    return None


def build_asteroid_batch(
    records: list[Mapping[str, object]],
    gm_values: Mapping[int, float],
    *,
    albedo: float,
    density: float,
    unknown_mass: float,
) -> AsteroidBatch:
    size = len(records)
    values = {
        key: np.asarray([float(record[key]) for record in records], dtype=np.float64)
        for key in ("e", "a", "i", "Node", "Peri", "M", "Epoch")
    }

    mean_motion_values = []
    solar_mu = gm_values[10] * 1.0e9
    for record, semimajor_axis in zip(records, values["a"]):
        mean_motion = _finite_float(record, "n")
        if mean_motion is None or mean_motion <= 0.0:
            axis_metres = abs(semimajor_axis) * AU_METERS
            mean_motion = math.degrees(math.sqrt(solar_mu / axis_metres**3)) * SECONDS_PER_DAY
        mean_motion_values.append(mean_motion)

    mass = np.empty(size, dtype=np.float64)
    mass_sources: list[str] = []
    for index, record in enumerate(records):
        gm = known_asteroid_gm(asteroid_number(record), gm_values)
        if gm is not None:
            mass[index] = gm * 1.0e9 / GRAVITATIONAL_CONSTANT
            mass_sources.append("JPL GM")
            continue
        absolute_magnitude = _finite_float(record, "H")
        if absolute_magnitude is None:
            mass[index] = unknown_mass
            mass_sources.append("configured fallback")
            continue
        diameter_km = 1329.0 / math.sqrt(albedo) * 10.0 ** (-absolute_magnitude / 5.0)
        mass[index] = density * (math.pi / 6.0) * (diameter_km * 1_000.0) ** 3
        mass_sources.append("estimated from H")

    return AsteroidBatch(
        records=records,
        eccentricity=values["e"],
        semimajor_axis_au=values["a"],
        inclination=np.radians(values["i"]),
        ascending_node=np.radians(values["Node"]),
        argument_of_perihelion=np.radians(values["Peri"]),
        mean_anomaly=np.radians(values["M"]),
        mean_motion=np.radians(np.asarray(mean_motion_values, dtype=np.float64)),
        orbit_epoch_jd=values["Epoch"],
        mass=mass,
        mass_sources=mass_sources,
    )


def _solve_elliptic_kepler(mean_anomaly: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    anomaly = np.where(eccentricity < 0.8, mean_anomaly, math.pi)
    for _ in range(32):
        correction = (
            anomaly - eccentricity * np.sin(anomaly) - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(anomaly))
        anomaly -= correction
        if correction.size == 0 or np.max(np.abs(correction)) < 1.0e-13:
            return anomaly
    raise ValueError("elliptic Kepler solver did not converge")


def _solve_hyperbolic_kepler(mean_anomaly: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    anomaly = np.arcsinh(mean_anomaly / eccentricity)
    for _ in range(48):
        correction = (
            eccentricity * np.sinh(anomaly) - anomaly - mean_anomaly
        ) / (eccentricity * np.cosh(anomaly) - 1.0)
        anomaly -= correction
        if correction.size == 0 or np.max(np.abs(correction)) < 1.0e-13:
            return anomaly
    raise ValueError("hyperbolic Kepler solver did not converge")


def propagate_asteroid_batch(
    batch: AsteroidBatch, target_jd: float, sun_barycentric_state: np.ndarray
) -> np.ndarray:
    size = batch.eccentricity.size
    xp = np.empty(size, dtype=np.float64)
    yp = np.empty(size, dtype=np.float64)
    vxp = np.empty(size, dtype=np.float64)
    vyp = np.empty(size, dtype=np.float64)
    elapsed_days = target_jd - batch.orbit_epoch_jd
    mean_anomaly = batch.mean_anomaly + batch.mean_motion * elapsed_days

    elliptic = batch.eccentricity < 1.0
    if np.any(elliptic):
        eccentricity = batch.eccentricity[elliptic]
        anomaly = _solve_elliptic_kepler(
            np.mod(mean_anomaly[elliptic], 2.0 * math.pi), eccentricity
        )
        axis = batch.semimajor_axis_au[elliptic] * AU_METERS
        root = np.sqrt(1.0 - eccentricity**2)
        denominator = 1.0 - eccentricity * np.cos(anomaly)
        anomaly_rate = (
            batch.mean_motion[elliptic] / SECONDS_PER_DAY / denominator
        )
        xp[elliptic] = axis * (np.cos(anomaly) - eccentricity)
        yp[elliptic] = axis * root * np.sin(anomaly)
        vxp[elliptic] = -axis * np.sin(anomaly) * anomaly_rate
        vyp[elliptic] = axis * root * np.cos(anomaly) * anomaly_rate

    hyperbolic = ~elliptic
    if np.any(hyperbolic):
        eccentricity = batch.eccentricity[hyperbolic]
        anomaly = _solve_hyperbolic_kepler(mean_anomaly[hyperbolic], eccentricity)
        axis = -batch.semimajor_axis_au[hyperbolic] * AU_METERS
        root = np.sqrt(eccentricity**2 - 1.0)
        denominator = eccentricity * np.cosh(anomaly) - 1.0
        anomaly_rate = (
            batch.mean_motion[hyperbolic] / SECONDS_PER_DAY / denominator
        )
        xp[hyperbolic] = axis * (eccentricity - np.cosh(anomaly))
        yp[hyperbolic] = axis * root * np.sinh(anomaly)
        vxp[hyperbolic] = -axis * np.sinh(anomaly) * anomaly_rate
        vyp[hyperbolic] = axis * root * np.cosh(anomaly) * anomaly_rate

    cosine_node = np.cos(batch.ascending_node)
    sine_node = np.sin(batch.ascending_node)
    cosine_perihelion = np.cos(batch.argument_of_perihelion)
    sine_perihelion = np.sin(batch.argument_of_perihelion)
    cosine_inclination = np.cos(batch.inclination)
    sine_inclination = np.sin(batch.inclination)

    r11 = cosine_node * cosine_perihelion - sine_node * sine_perihelion * cosine_inclination
    r12 = -cosine_node * sine_perihelion - sine_node * cosine_perihelion * cosine_inclination
    r21 = sine_node * cosine_perihelion + cosine_node * sine_perihelion * cosine_inclination
    r22 = -sine_node * sine_perihelion + cosine_node * cosine_perihelion * cosine_inclination
    r31 = sine_perihelion * sine_inclination
    r32 = cosine_perihelion * sine_inclination

    x_ecliptic = r11 * xp + r12 * yp
    y_ecliptic = r21 * xp + r22 * yp
    z_ecliptic = r31 * xp + r32 * yp
    vx_ecliptic = r11 * vxp + r12 * vyp
    vy_ecliptic = r21 * vxp + r22 * vyp
    vz_ecliptic = r31 * vxp + r32 * vyp

    cosine_obliquity = math.cos(J2000_OBLIQUITY)
    sine_obliquity = math.sin(J2000_OBLIQUITY)
    state = np.column_stack(
        (
            x_ecliptic,
            cosine_obliquity * y_ecliptic - sine_obliquity * z_ecliptic,
            sine_obliquity * y_ecliptic + cosine_obliquity * z_ecliptic,
            vx_ecliptic,
            cosine_obliquity * vy_ecliptic - sine_obliquity * vz_ecliptic,
            sine_obliquity * vy_ecliptic + cosine_obliquity * vz_ecliptic,
        )
    )
    state += np.asarray(sun_barycentric_state, dtype=np.float64)
    return state


def asteroid_identity(record: Mapping[str, object]) -> tuple[str, str]:
    number = asteroid_number(record)
    designation = str(record.get("Principal_desig") or "").strip()
    name = str(record.get("Name") or "").strip()
    identifier = str(number) if number is not None else designation
    display_name = name or designation or identifier or "unnamed asteroid"
    return identifier, display_name


def write_manifest_row(
    writer: csv.writer,
    index: int,
    category: str,
    identifier: str,
    name: str,
    mass: float,
    mass_source: str,
    orbit_epoch: object = "",
) -> None:
    writer.writerow(
        (index, category, identifier, name, f"{mass:.17g}", mass_source, orbit_epoch)
    )


def ensure_outputs_available(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists (use --overwrite): {joined}")


def build_datasets(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_a = args.output_dir / f"{args.prefix}_epoch_a.bin"
    output_b = args.output_dir / f"{args.prefix}_epoch_b.bin"
    manifest_path = args.output_dir / f"{args.prefix}_objects.csv.gz"
    metadata_path = args.output_dir / f"{args.prefix}_metadata.json"
    ensure_outputs_available(
        (output_a, output_b, manifest_path, metadata_path), args.overwrite
    )

    mpcorb_path = args.mpcorb_file or (args.cache_dir / "mpcorb_extended.json.gz")
    if args.mpcorb_file is None:
        print(f"Fetching MPC asteroid catalogue into {mpcorb_path}...")
        fetch_to_cache(
            MPCORB_URL,
            mpcorb_path,
            timeout=args.http_timeout,
            retries=args.http_retries,
            offline=args.offline,
            refresh=args.refresh,
        )
    elif not mpcorb_path.is_file():
        raise FileNotFoundError(mpcorb_path)

    gm_path = args.gm_file or (args.cache_dir / "gm_Horizons.pck")
    if args.gm_file is None:
        fetch_to_cache(
            HORIZONS_GM_URL,
            gm_path,
            timeout=args.http_timeout,
            retries=args.http_retries,
            offline=args.offline,
            refresh=args.refresh,
        )
    elif not gm_path.is_file():
        raise FileNotFoundError(gm_path)
    gm_values = parse_gm_pck(gm_path.read_text(encoding="utf-8"))

    major_body_path = args.major_bodies_file or (
        args.cache_dir / "horizons_major_bodies.txt"
    )
    if args.major_bodies_file is None:
        major_body_text = fetch_api_text(
            {
                "format": "text",
                "COMMAND": "MB",
                "MAKE_EPHEM": "NO",
                "OBJ_DATA": "YES",
            },
            major_body_path,
            timeout=args.http_timeout,
            retries=args.http_retries,
            offline=args.offline,
            refresh=args.refresh,
        )
    else:
        major_body_text = major_body_path.read_text(encoding="utf-8")
    major_bodies = parse_major_body_index(major_body_text)
    epochs = (args.epoch_a, args.epoch_b)
    major_bodies, major_states, major_failures = load_major_body_states(
        major_bodies, epochs, args
    )

    partial_manifest = manifest_path.with_name(manifest_path.name + ".part")
    partial_manifest.unlink(missing_ok=True)
    skipped_invalid_orbits = 0
    asteroid_count = 0
    known_gm_count = 0
    estimated_mass_count = 0
    fallback_mass_count = 0
    try:
        with ParticleDatasetWriter(output_a) as writer_a, ParticleDatasetWriter(
            output_b
        ) as writer_b, gzip.open(
            partial_manifest, "wt", encoding="utf-8", newline=""
        ) as manifest_file:
            manifest = csv.writer(manifest_file)
            manifest.writerow(
                (
                    "index",
                    "category",
                    "identifier",
                    "name",
                    "mass_kg",
                    "mass_source",
                    "orbit_epoch_jd_tdb",
                )
            )

            major_mass = np.asarray(
                [
                    gm_values.get(body.spkid, 0.0)
                    * 1.0e9
                    / GRAVITATIONAL_CONSTANT
                    for body in major_bodies
                ],
                dtype=np.float64,
            )
            major_type = np.asarray(
                [
                    {
                        "star": ParticleType.STAR,
                        "planet": ParticleType.PLANET,
                        "moon": ParticleType.MOON,
                    }[body.category]
                    for body in major_bodies
                ],
                dtype=np.uint8,
            )
            for epoch_index, writer in enumerate((writer_a, writer_b)):
                state = np.vstack(
                    [major_states[body.spkid][epoch_index] for body in major_bodies]
                )
                writer.append(major_mass, state, major_type)
            for index, (body, mass) in enumerate(zip(major_bodies, major_mass)):
                source = "JPL GM" if body.spkid in gm_values else "massless (unknown GM)"
                write_manifest_row(
                    manifest,
                    index,
                    body.category,
                    str(body.spkid),
                    body.name,
                    float(mass),
                    source,
                )

            sun_state_a = major_states[10][0]
            sun_state_b = major_states[10][1]
            pending: list[Mapping[str, object]] = []

            def flush_pending() -> None:
                nonlocal asteroid_count, known_gm_count, estimated_mass_count, fallback_mass_count
                if not pending:
                    return
                records = list(pending)
                pending.clear()
                batch = build_asteroid_batch(
                    records,
                    gm_values,
                    albedo=args.asteroid_albedo,
                    density=args.asteroid_density,
                    unknown_mass=args.unknown_asteroid_mass,
                )
                state_a = propagate_asteroid_batch(
                    batch, args.epoch_a.jd_tdb, sun_state_a
                )
                state_b = propagate_asteroid_batch(
                    batch, args.epoch_b.jd_tdb, sun_state_b
                )
                finite = (
                    np.isfinite(batch.mass)
                    & np.all(np.isfinite(state_a), axis=1)
                    & np.all(np.isfinite(state_b), axis=1)
                )
                if not np.all(finite):
                    raise ValueError("asteroid propagation generated non-finite data")
                asteroid_type = np.full(
                    len(records), ParticleType.ASTEROID, dtype=np.uint8
                )
                writer_a.append(batch.mass, state_a, asteroid_type)
                writer_b.append(batch.mass, state_b, asteroid_type)
                base_index = len(major_bodies) + asteroid_count
                for offset, (record, mass, source) in enumerate(
                    zip(batch.records, batch.mass, batch.mass_sources)
                ):
                    identifier, name = asteroid_identity(record)
                    write_manifest_row(
                        manifest,
                        base_index + offset,
                        "asteroid",
                        identifier,
                        name,
                        float(mass),
                        source,
                        record.get("Epoch", ""),
                    )
                asteroid_count += len(records)
                known_gm_count += batch.mass_sources.count("JPL GM")
                estimated_mass_count += batch.mass_sources.count("estimated from H")
                fallback_mass_count += batch.mass_sources.count("configured fallback")
                if asteroid_count % 100_000 < len(records):
                    print(f"  MPCORB: {asteroid_count:,} asteroids written")

            with gzip.open(mpcorb_path, "rt", encoding="utf-8") as catalogue:
                for value in iter_json_array(catalogue):
                    if not isinstance(value, Mapping) or not valid_asteroid_record(value):
                        skipped_invalid_orbits += 1
                        continue
                    pending.append(value)
                    if len(pending) >= args.batch_size:
                        flush_pending()
                    if (
                        args.max_asteroids is not None
                        and asteroid_count + len(pending) >= args.max_asteroids
                    ):
                        if asteroid_count + len(pending) > args.max_asteroids:
                            del pending[args.max_asteroids - asteroid_count :]
                        break
                flush_pending()

            writer_a.finalize()
            writer_b.finalize()
        os.replace(partial_manifest, manifest_path)
    finally:
        partial_manifest.unlink(missing_ok=True)

    particle_count = len(major_bodies) + asteroid_count
    metadata: dict[str, object] = {
        "format": (
            "AMSC-Nbody planar little-endian: uint64 count header, seven float64 "
            "blocks, then one uint8 particle-type block"
        ),
        "particle_count": particle_count,
        "major_body_count": len(major_bodies),
        "asteroid_count": asteroid_count,
        "epoch_a": {"input": args.epoch_a.text, "jd_tdb": args.epoch_a.jd_tdb},
        "epoch_b": {"input": args.epoch_b.text, "jd_tdb": args.epoch_b.jd_tdb},
        "output_a": str(output_a),
        "output_b": str(output_b),
        "manifest": str(manifest_path),
        "coordinates": "Solar-System-barycentric ICRF Cartesian",
        "units": {"mass": "kg", "position": "m", "velocity": "m/s"},
        "sources": {
            "major_body_states": HORIZONS_API_URL,
            "major_body_gm": HORIZONS_GM_URL,
            "asteroid_orbits": MPCORB_URL,
        },
        "asteroid_model": (
            "MPC heliocentric J2000-ecliptic osculating elements propagated independently "
            "with a two-body Kepler model, rotated to ICRF, then translated by the "
            "HORIZONS barycentric Sun state"
        ),
        "mass_model": {
            "known_gm_count": known_gm_count,
            "estimated_from_H_count": estimated_mass_count,
            "fallback_count": fallback_mass_count,
            "assumed_albedo": args.asteroid_albedo,
            "assumed_density_kg_m3": args.asteroid_density,
            "unknown_asteroid_mass_kg": args.unknown_asteroid_mass,
            "unknown_major_body_gm_policy": "massless test particle",
        },
        "skipped_invalid_asteroid_orbits": skipped_invalid_orbits,
        "skipped_major_bodies": major_failures,
    }
    partial_metadata = metadata_path.with_name(metadata_path.name + ".part")
    try:
        partial_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial_metadata, metadata_path)
    finally:
        partial_metadata.unlink(missing_ok=True)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        metadata = build_datasets(args)
    except (DownloadError, FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"solar-system dataset build failed: {error}", file=sys.stderr)
        return 1

    print("Solar-system datasets are ready:")
    print(f"  epoch A: {metadata['output_a']}")
    print(f"  epoch B: {metadata['output_b']}")
    print(f"  particles: {metadata['particle_count']:,}")
    print(f"  manifest: {metadata['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
