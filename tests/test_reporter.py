"""End-to-end tests for the OpenMM reporter."""

from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

from chillplus import ChillPlusReporter

BOX_HEADER = (
    "#Step\tTime(ns)\tTemperature(K)\tIc\tIh\tMax Cluster\t"
    "Ic ratio\tIh ratio\tInterfacial Ice\tClathrate\t"
    "Interfacial Clathrate\tOther"
)


def _build_simulation(
    positions,
    box_size=None,
    residue_name="HOH",
    element=app.element.oxygen,
    periodic=True,
):
    """Build a minimal O-only OpenMM system for reporter tests."""
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(len(positions)):
        residue = topology.addResidue(residue_name, chain)
        topology.addAtom("O", element, residue)

    box_vectors = None
    if periodic:
        if box_size is None:
            raise ValueError("box_size is required for a periodic test system")
        lengths = np.asarray(box_size, dtype=float) / 10.0  # angstrom -> nm
        box_vectors = (
            mm.Vec3(lengths[0], 0.0, 0.0),
            mm.Vec3(0.0, lengths[1], 0.0),
            mm.Vec3(0.0, 0.0, lengths[2]),
        )
        topology.setPeriodicBoxVectors(box_vectors)

    system = mm.System()
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)
    for _ in positions:
        system.addParticle(16.0)

    force = mm.NonbondedForce()
    if periodic:
        force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(0.5 * unit.nanometer)
    else:
        force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for _ in positions:
        force.addParticle(0.0, 0.3, 0.1)
    system.addForce(force)

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picosecond,
    )
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(np.asarray(positions) / 10.0)
    return simulation


def test_reporter_writes_counts_and_indices(tmp_path, cubic_ice):
    positions, box_size = cubic_ice
    simulation = _build_simulation(positions, box_size)
    output = tmp_path / "structures.dat"

    with ChillPlusReporter(output, interval=2, need_index=True) as reporter:
        simulation.reporters.append(reporter)
        simulation.step(4)

    assert reporter.max_cluster == 64
    assert reporter.last_counts["Ic"] == 64
    assert reporter.last_counts["Other"] == 0
    assert reporter.last_counts["Ic ratio"] == "1.00"

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == BOX_HEADER
    assert len(lines) == 3
    for line in lines[1:]:
        fields = line.split("\t")
        assert len(fields) == 12
        assert fields[0] in {"2", "4"}
        assert fields[3] == "64"
        assert fields[4] == "0"
        assert fields[5] == "64"
        assert fields[6] == "1.00"

    index_path = tmp_path / "structures_index.dat"
    index_lines = index_path.read_text(encoding="utf-8").splitlines()
    assert index_lines[0] == "#Step\tOxygen index"
    assert len(index_lines) == 3
    for line in index_lines[1:]:
        step, indices = line.split("\t", maxsplit=1)
        assert step in {"2", "4"}
        assert indices.split() == [str(i) for i in range(64)]


def test_reporter_respects_water_name(tmp_path, cubic_ice):
    positions, box_size = cubic_ice
    simulation = _build_simulation(positions, box_size, residue_name="WAT")
    output = tmp_path / "wat.dat"

    with ChillPlusReporter(output, interval=1, water_name="WAT") as reporter:
        simulation.reporters.append(reporter)
        simulation.step(1)

    assert reporter.max_cluster == 64
    assert output.exists()
    assert not (tmp_path / "wat_index.dat").exists()


def test_reporter_accepts_explicit_group(tmp_path, cubic_ice):
    positions, box_size = cubic_ice
    simulation = _build_simulation(positions, box_size)
    output = tmp_path / "group.dat"

    with ChillPlusReporter(output, interval=1, group=range(64)) as reporter:
        simulation.reporters.append(reporter)
        simulation.step(1)

    assert reporter.last_counts["Ic"] == 64


def test_reporter_without_ice(tmp_path):
    positions = np.array([[1.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    box_size = np.array([20.0, 20.0, 20.0])
    simulation = _build_simulation(positions, box_size)
    output = tmp_path / "melted.dat"

    with ChillPlusReporter(output, interval=1, need_index=True) as reporter:
        simulation.reporters.append(reporter)
        simulation.step(1)

    assert reporter.max_cluster == 0
    assert reporter.last_counts["Ic ratio"] == "0.00"
    assert reporter.last_counts["Other"] == 2

    last_line = output.read_text(encoding="utf-8").splitlines()[-1]
    assert last_line.split("\t")[5] == "0"

    index_lines = (
        (tmp_path / "melted_index.dat").read_text(encoding="utf-8").splitlines()
    )
    assert index_lines[-1] == "1\t"


def test_reporter_rejects_triclinic_box(tmp_path, cubic_ice):
    positions, box_size = cubic_ice
    simulation = _build_simulation(positions, box_size)
    lengths = box_size / 10.0
    triclinic = (
        mm.Vec3(lengths[0], 0.0, 0.0),
        mm.Vec3(0.1, lengths[1], 0.0),
        mm.Vec3(0.0, 0.0, lengths[2]),
    )
    simulation.context.setPeriodicBoxVectors(*triclinic)
    output = tmp_path / "triclinic.dat"

    with ChillPlusReporter(output, interval=1) as reporter:
        simulation.reporters.append(reporter)
        with pytest.raises(NotImplementedError, match="orthorhombic"):
            simulation.step(1)


def test_reporter_validates_arguments(tmp_path):
    with pytest.raises(ValueError, match="interval"):
        ChillPlusReporter(tmp_path / "x.dat", 0)
    with pytest.raises(ValueError, match="cluster"):
        ChillPlusReporter(tmp_path / "x.dat", 1, cluster=["bogus"])
    with pytest.raises(ValueError, match="duplicate"):
        ChillPlusReporter(tmp_path / "x.dat", 1, cluster=["Ic", "Ic"])
    with pytest.raises(ValueError, match="group"):
        ChillPlusReporter(tmp_path / "x.dat", 1, group=[])
    with pytest.raises(ValueError, match="group"):
        ChillPlusReporter(tmp_path / "x.dat", 1, group=[0, 0])


def test_reporter_rejects_nonperiodic_system(tmp_path):
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.75, 0.0, 0.0], [0.0, 2.75, 0.0], [0.0, 0.0, 2.75]]
    )
    simulation = _build_simulation(positions, periodic=False)
    output = tmp_path / "nonperiodic.dat"

    with ChillPlusReporter(output, interval=1) as reporter:
        simulation.reporters.append(reporter)
        with pytest.raises(ValueError, match="periodic boundary"):
            simulation.step(1)


def test_reporter_selects_oxygen_without_element(tmp_path, cubic_ice):
    positions, box_size = cubic_ice
    simulation = _build_simulation(positions, box_size, element=None)
    output = tmp_path / "noelement.dat"

    with ChillPlusReporter(output, interval=1) as reporter:
        simulation.reporters.append(reporter)
        simulation.step(1)

    assert reporter.last_counts["Ic"] == 64


def test_reporter_appends_dat_suffix(tmp_path, cubic_ice):
    positions, box_size = cubic_ice
    simulation = _build_simulation(positions, box_size)
    output = tmp_path / "structures"

    with ChillPlusReporter(output, interval=1) as reporter:
        simulation.reporters.append(reporter)
        simulation.step(1)

    assert reporter.file_path == tmp_path / "structures.dat"
    assert reporter.file_path.exists()


def test_reporter_validates_constructor_edge_cases(tmp_path, cubic_ice):
    positions, box_size = cubic_ice
    simulation = _build_simulation(positions, box_size)
    output = tmp_path / "invalid.dat"

    with pytest.raises(ValueError, match="cut_off"):
        ChillPlusReporter(output, 1, cut_off=float("nan"))
    with pytest.raises(ValueError, match="cluster_cut_off"):
        ChillPlusReporter(output, 1, cluster_cut_off=float("inf"))
    with pytest.raises(ValueError, match="integer atom indices"):
        ChillPlusReporter(output, 1, group=[0.9, 1.1])
    with pytest.raises(ValueError, match="sequence"):
        ChillPlusReporter(output, 1, cluster="Ic")
    with pytest.raises(ValueError, match="strings"):
        ChillPlusReporter(output, 1, cluster=[1])
    with pytest.raises(TypeError, match="OpenMM unit"):
        ChillPlusReporter(output, 1, time_unit="ns")
    assert not output.exists()

    # An integral float array is still a valid index list.
    with ChillPlusReporter(output, 1, group=[0.0, 1.0]) as reporter:
        simulation.reporters.append(reporter)
        simulation.step(1)
    assert reporter.ids is not None
    np.testing.assert_array_equal(reporter.ids, [0, 1])


def test_reporter_rejects_out_of_range_group(tmp_path, cubic_ice):
    positions, box_size = cubic_ice
    simulation = _build_simulation(positions, box_size)
    output = tmp_path / "badgroup.dat"

    with ChillPlusReporter(output, interval=1, group=[0, 999]) as reporter:
        simulation.reporters.append(reporter)
        with pytest.raises(ValueError, match="outside the topology"):
            simulation.step(1)


def test_reporter_close_is_idempotent(tmp_path):
    path = Path(tmp_path) / "closed.dat"
    reporter = ChillPlusReporter(path, interval=1)
    reporter.close()
    reporter.close()
    assert reporter.file.closed
