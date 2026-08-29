"""Stable particle-type values shared by the Python dataset generators."""

from enum import IntEnum


class ParticleType(IntEnum):
    UNKNOWN = 0
    STAR = 1
    PLANET = 2
    MOON = 3
    ASTEROID = 4


VALID_PARTICLE_TYPES = frozenset(ParticleType)
