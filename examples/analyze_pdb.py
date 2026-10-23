#!/usr/bin/env python3
"""Classify the oxygen atoms of a PDB file with CHILL+.

Run from the repository root::

    python examples/analyze_pdb.py

The default input is the bundled 195 K configuration, which contains 4096
water molecules and reproduces the reference CHILL+ counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from openmm.app import PDBFile
from openmm.unit import angstrom

from chillplus import classify, largest_cluster

DEFAULT_PDB = Path(__file__).resolve().parent / "data" / "ice_water_4096_195K.pdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdb",
        nargs="?",
        type=Path,
        default=DEFAULT_PDB,
        help=(
            "PDB file containing water oxygens "
            "(default: bundled ice_water_4096_195K.pdb)"
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=3.5,
        help="O-O neighbor cutoff in angstrom (default: 3.5)",
    )
    parser.add_argument(
        "--cluster-cutoff",
        type=float,
        default=3.2,
        help="ice-nucleus connection cutoff in angstrom (default: 3.2)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdb = PDBFile(str(args.pdb))
    topology = pdb.topology

    oxygen_atoms = [
        atom
        for atom in topology.atoms()
        if (atom.element is not None and atom.element.symbol == "O") or atom.name == "O"
    ]
    if not oxygen_atoms:
        raise SystemExit(f"no oxygen atoms found in {args.pdb}")

    positions = pdb.positions.value_in_unit(angstrom)
    positions = np.asarray(positions)[[atom.index for atom in oxygen_atoms]]

    box_vectors = topology.getPeriodicBoxVectors()
    if box_vectors is None:
        raise SystemExit("the PDB file does not define a periodic box")
    box = np.asarray(box_vectors.value_in_unit(angstrom))
    if not np.allclose(box, np.diag(np.diag(box)), atol=1e-6):
        raise SystemExit(
            "this example only supports orthorhombic boxes; CHILL+ in this "
            "package assumes an orthorhombic periodic box"
        )
    box_size = np.diag(box)

    start = perf_counter()
    counts, labels = classify(positions, box_size, cut_off=args.cutoff, l=3)
    elapsed = perf_counter() - start

    print(f"Input: {args.pdb}")
    print(f"Oxygen atoms: {len(positions)}")
    print("Structure counts:")
    for name, count in counts.items():
        print(f"  {name:<22} {count:>6d}")

    ice = np.isin(labels, ("Ic", "Ih"))
    if np.any(ice):
        _, sizes = largest_cluster(
            positions[ice], box_size, cut_off=args.cluster_cutoff
        )
        print(f"Ice atoms: {int(np.count_nonzero(ice))}")
        print(f"Largest ice cluster: {sizes.get(1, 0)}")
        print(f"Number of ice clusters: {len(sizes)}")
    else:
        print("No ice atoms detected.")

    print(f"CHILL+ time: {elapsed:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
