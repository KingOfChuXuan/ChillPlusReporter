#!/usr/bin/env python3
"""Run a short OpenMM simulation with :class:`ChillPlusReporter`.

The system is a toy cubic-ice lattice of oxygen atoms, not a physical water
model.  It only demonstrates how to attach the reporter to a simulation and
what the output files look like.  Run from the repository root::

    python examples/reporter_demo.py --output demo.dat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

from chillplus import ChillPlusReporter


def cubic_ice(repeats: int = 2, lattice: float = 6.35):
    """Return ideal cubic-ice oxygen positions (angstrom) and box lengths."""
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
    return np.asarray(positions), np.full(3, repeats * lattice)


def build_simulation(positions, box_size):
    """Build a minimal O-only OpenMM simulation."""
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(len(positions)):
        residue = topology.addResidue("HOH", chain)
        topology.addAtom("O", app.element.oxygen, residue)

    lengths = np.asarray(box_size) / 10.0  # angstrom -> nm
    box_vectors = (
        mm.Vec3(lengths[0], 0.0, 0.0),
        mm.Vec3(0.0, lengths[1], 0.0),
        mm.Vec3(0.0, 0.0, lengths[2]),
    )
    topology.setPeriodicBoxVectors(box_vectors)

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    for _ in positions:
        system.addParticle(16.0)

    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(0.5 * unit.nanometer)
    for _ in positions:
        force.addParticle(0.0, 0.3, 0.1)
    system.addForce(force)

    integrator = mm.LangevinMiddleIntegrator(
        250 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picosecond,
    )
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions / 10.0)
    return simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo.dat"),
        help="output file for structure counts (default: demo.dat)",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="number of MD steps (default: 50)"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="also write the oxygen indices of the largest ice nucleus",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    positions, box_size = cubic_ice()
    simulation = build_simulation(positions, box_size)

    reporter = ChillPlusReporter(
        args.output,
        interval=10,
        need_index=args.index,
    )
    simulation.reporters.append(reporter)

    print(f"Running {args.steps} steps with {len(positions)} oxygen atoms...")
    simulation.step(args.steps)
    reporter.close()

    print(f"Largest ice nucleus: {reporter.max_cluster}")
    print(f"Structure counts: {reporter.last_counts}")
    print(f"Wrote {reporter.file_path}")
    if reporter.index_path is not None:
        print(f"Wrote {reporter.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
