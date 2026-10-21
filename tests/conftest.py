"""Shared fixtures for the ChillPlusReporter test suite."""

import numpy as np
import pytest


def _cubic_ice(repeats: int = 2, lattice: float = 6.35):
    """Build an ideal cubic-ice (diamond) oxygen lattice in angstrom."""
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ]
    )
    positions = []
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                positions.extend((basis + [i, j, k]) * lattice)
    box_size = np.full(3, repeats * lattice, dtype=float)
    return np.asarray(positions), box_size


@pytest.fixture(scope="session")
def cubic_ice():
    """A periodic 2x2x2 cubic-ice supercell with 64 oxygen atoms."""
    return _cubic_ice()
