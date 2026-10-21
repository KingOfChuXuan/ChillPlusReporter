"""Regression tests on the bundled 195 K water/ice configuration."""

from pathlib import Path

import numpy as np

from chillplus import classify, largest_cluster

DATA = Path(__file__).resolve().parents[1] / "examples" / "data" / "195K.pdb"


def _oxygen_positions():
    from openmm.app import PDBFile
    from openmm.unit import angstrom

    pdb = PDBFile(str(DATA))
    topology = pdb.topology
    indices = [
        atom.index
        for atom in topology.atoms()
        if atom.element is not None and atom.element.symbol == "O"
    ]
    positions = np.array(pdb.positions.value_in_unit(angstrom))[indices]
    box_vectors = np.array(topology.getPeriodicBoxVectors().value_in_unit(angstrom))
    return positions, np.diag(box_vectors)


def test_reference_configuration_counts():
    positions, box_size = _oxygen_positions()
    assert positions.shape == (4096, 3)

    counts, labels = classify(positions, box_size)
    assert counts == {
        "Ic": 1109,
        "Ih": 544,
        "Interfacial Ice": 1047,
        "Clathrate": 5,
        "Interfacial Clathrate": 62,
        "Other": 1329,
    }
    assert sum(counts.values()) == len(positions)

    ice = np.isin(labels, ("Ic", "Ih"))
    assert np.count_nonzero(ice) == 1653

    _, sizes = largest_cluster(positions[ice], box_size, cut_off=3.2)
    assert sizes[1] == 1561
    assert len(sizes) == 58
