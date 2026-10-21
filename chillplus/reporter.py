"""An OpenMM reporter for CHILL+ water-structure analysis.

The reporter evaluates :func:`chillplus.core.classify` on selected frames and
writes the population of each structure type, the size and composition of the
largest ice nucleus, and optionally the oxygen indices of that nucleus.

Only oxygen atoms are analysed.  The system must use an orthorhombic periodic
box; triclinic boxes are rejected with :class:`NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import numpy as np
from openmm.app import Element
from openmm.unit import Unit, angstrom, kelvin, nanosecond

from .core import (
    DEFAULT_CLUSTER_CUTOFF,
    DEFAULT_CUTOFF,
    ICE_STRUCTURES,
    STRUCTURE_TYPES,
    classify,
    largest_cluster,
)


def _output_path(file: str | Path) -> Path:
    """Return the output path, appending ``.dat`` when it has no suffix."""
    path = Path(file)
    if not path.suffix:
        path = path.with_suffix(".dat")
    return path


def _positive_float(name: str, value: float) -> float:
    """Return ``value`` as a finite positive float, or raise ValueError."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


class ChillPlusReporter:
    """OpenMM reporter that records CHILL+ structure populations.

    Parameters
    ----------
    file : str or pathlib.Path
        Output file for the per-frame structure counts.  A ``.dat`` suffix is
        appended when the path has no suffix.
    interval : int
        Report interval in simulation steps.
    need_index : bool, optional
        If ``True``, also write the oxygen atom indices belonging to the
        largest ice nucleus.  The indices are written to
        ``<stem>_index<suffix>``.
    group : sequence of int, optional
        Atom indices to analyse instead of all oxygen atoms of residues named
        ``water_name``.  The indices must refer to oxygen atoms.  Selected and
        unselected water molecules should not be in contact, because the
        neighbour count of a selected atom is computed only from the selected
        subset.
    cut_off : float, optional
        O-O cutoff for the first coordination shell, in angstrom.  Default is
        3.5.
    l : int, optional
        Spherical-harmonic degree.  Default is 3.
    cluster : sequence of str, optional
        Structure labels that count as the ice nucleus.  Default is
        ``("Ic", "Ih")``.  Labels must be elements of
        :data:`chillplus.core.STRUCTURE_TYPES`.
    cluster_cut_off : float, optional
        Distance cutoff used to connect atoms into nuclei, in angstrom.
        Default is 3.2.
    water_name : str, optional
        Residue name of water molecules in the topology.  Oxygen atoms are
        matched by element, or by atom name ``"O"`` when element information
        is absent.  Default is ``"HOH"``.
    time_unit : openmm.unit.Unit, optional
        Unit used for the time column.  Default is ``nanosecond``.

    Notes
    -----
    The simulation must use periodic boundary conditions.  The reporter
    requests wrapped positions from OpenMM and raises :class:`ValueError` for
    non-periodic systems, because OpenMM would otherwise supply a default 2 nm
    box.  Call :meth:`close` when the simulation is finished (or use the
    reporter as a context manager) to flush and close the output files.
    """

    def __init__(
        self,
        file: str | Path,
        interval: int,
        need_index: bool = False,
        group: Sequence[int] | None = None,
        cut_off: float = DEFAULT_CUTOFF,
        l: int = 3,
        cluster: Sequence[str] | None = None,
        cluster_cut_off: float = DEFAULT_CLUSTER_CUTOFF,
        water_name: str = "HOH",
        time_unit: Unit = nanosecond,
    ) -> None:
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, np.integer))
            or interval <= 0
        ):
            raise ValueError("interval must be a positive integer")
        if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or l < 0:
            raise ValueError("l must be a non-negative integer")
        cut_off = _positive_float("cut_off", cut_off)
        cluster_cut_off = _positive_float("cluster_cut_off", cluster_cut_off)
        if not hasattr(time_unit, "get_symbol"):
            raise TypeError(
                "time_unit must be an OpenMM unit, e.g. openmm.unit.nanosecond"
            )
        time_symbol = time_unit.get_symbol()

        if isinstance(cluster, str):
            raise ValueError(
                "cluster must be a sequence of structure labels, not a string"
            )
        cluster_names = list(ICE_STRUCTURES) if cluster is None else list(cluster)
        if not cluster_names:
            raise ValueError("cluster must contain at least one structure label")
        if not all(isinstance(name, str) for name in cluster_names):
            raise ValueError("cluster labels must be strings")
        if len(set(cluster_names)) != len(cluster_names):
            raise ValueError("cluster contains duplicate labels")
        unknown = sorted(set(cluster_names) - set(STRUCTURE_TYPES))
        if unknown:
            raise ValueError(
                f"unknown cluster label(s): {', '.join(unknown)}; "
                f"valid labels are {', '.join(STRUCTURE_TYPES)}"
            )

        if group is not None:
            group_array = np.asarray(group)
            if group_array.ndim != 1:
                raise ValueError("group must be a one-dimensional sequence")
            if group_array.size == 0:
                raise ValueError("group must contain at least one atom index")
            if not np.issubdtype(group_array.dtype, np.integer):
                if not np.issubdtype(group_array.dtype, np.floating):
                    raise ValueError("group must contain integer atom indices")
                if not np.all(np.isfinite(group_array)) or not np.all(
                    group_array == np.floor(group_array)
                ):
                    raise ValueError("group must contain integer atom indices")
            group_array = group_array.astype(np.int64, copy=False)
            if np.any(group_array < 0):
                raise ValueError("group contains a negative atom index")
            if np.unique(group_array).size != group_array.size:
                raise ValueError("group contains duplicate atom indices")
            self.ids: np.ndarray | None = group_array
        else:
            self.ids = None

        self.interval = int(interval)
        self.cut_off = cut_off
        self.l = int(l)
        self.cluster = tuple(cluster_names)
        self.cluster_cut_off = cluster_cut_off
        self.water_name = str(water_name)
        self.time_unit = time_unit
        self.need_index = bool(need_index)

        self.file_path = _output_path(file)
        self.index_path = (
            self.file_path.with_name(
                f"{self.file_path.stem}_index{self.file_path.suffix}"
            )
            if self.need_index
            else None
        )

        self.structures = self._column_names()
        self.max_cluster: int | None = None
        self.last_counts: dict[str, int | str] = {}
        self._periodic_checked = False
        self._closed = True

        self.file: TextIO = self.file_path.open("w", encoding="utf-8")
        try:
            self.index_file: TextIO | None = (
                self.index_path.open("w", encoding="utf-8")
                if self.index_path is not None
                else None
            )
        except Exception:
            self.file.close()
            raise
        self._closed = False

        header = [
            "#Step",
            f"Time({time_symbol})",
            "Temperature(K)",
            *self.structures,
        ]
        self.file.write("\t".join(header) + "\n")
        self.file.flush()
        if self.index_file is not None:
            self.index_file.write("#Step\tOxygen index\n")
            self.index_file.flush()

    def _column_names(self) -> list[str]:
        """Return the data columns in their output order."""
        cluster = list(self.cluster)
        columns = cluster + ["Max Cluster"]
        columns += [f"{name} ratio" for name in cluster]
        columns += [name for name in STRUCTURE_TYPES if name not in cluster]
        return columns

    def _select_oxygen_indices(self, topology) -> np.ndarray:
        """Select oxygen atoms of ``water_name`` residues from a topology."""
        oxygen = Element.getBySymbol("O")
        indices = [
            atom.index
            for atom in topology.atoms()
            if atom.residue.name == self.water_name
            and (atom.element == oxygen or (atom.element is None and atom.name == "O"))
        ]
        if not indices:
            raise ValueError(
                "no oxygen atoms found for residue name "
                f"{self.water_name!r}; set water_name to the residue name "
                "used by your topology"
            )
        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def _box_size(state) -> np.ndarray:
        """Return the diagonal of an orthorhombic periodic box in angstrom."""
        vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(angstrom)
        diagonal = np.diag(vectors)
        if np.any(diagonal <= 0.0):
            raise ValueError(
                "ChillPlusReporter requires a periodic system with a non-zero box"
            )
        if not np.allclose(vectors, np.diag(diagonal), atol=1e-6):
            raise NotImplementedError(
                "ChillPlusReporter only supports orthorhombic periodic boxes"
            )
        return diagonal

    @staticmethod
    def _temperature(simulation) -> float:
        """Return the integrator or context temperature in kelvin."""
        getter = getattr(simulation.integrator, "getTemperature", None)
        if callable(getter):
            try:
                return float(getter().value_in_unit(kelvin))
            except Exception:
                pass

        for name in simulation.context.getParameters():
            if "temperature" in name.lower():
                value = simulation.context.getParameter(name)
                if hasattr(value, "value_in_unit"):
                    value = value.value_in_unit(kelvin)
                return float(value)
        return float("nan")

    def _time(self, simulation) -> float:
        """Return the current simulation time in ``self.time_unit``."""
        getter = getattr(simulation.integrator, "getStepSize", None)
        if not callable(getter):
            return float("nan")
        try:
            step_size = getter()
            value = simulation.currentStep * step_size
            return float(value.value_in_unit(self.time_unit))
        except Exception:
            return float("nan")

    def describeNextReport(self, simulation):
        """Tell OpenMM that the next report needs wrapped positions."""
        steps = self.interval - simulation.currentStep % self.interval
        return steps, True, False, False, False, True

    def report(self, simulation, state) -> None:
        """Evaluate CHILL+ and write one row of output."""
        if self.ids is None:
            self.ids = self._select_oxygen_indices(simulation.topology)

        num_atoms = simulation.topology.getNumAtoms()
        if np.any(self.ids >= num_atoms):
            raise ValueError(
                "group contains an atom index outside the topology: "
                f"maximum index is {num_atoms - 1}"
            )

        if not self._periodic_checked:
            if not simulation.system.usesPeriodicBoundaryConditions():
                raise ValueError(
                    "ChillPlusReporter requires a system with periodic boundary "
                    "conditions; use chillplus.classify with an explicit box for "
                    "non-periodic systems"
                )
            self._periodic_checked = True

        positions = state.getPositions(asNumpy=True).value_in_unit(angstrom)
        positions = positions[self.ids]
        box_size = self._box_size(state)

        counts, labels = classify(positions, box_size, cut_off=self.cut_off, l=self.l)
        structure_counts: dict[str, int | str] = dict(counts)

        selected = np.isin(labels, self.cluster)
        if np.any(selected):
            cluster_labels, cluster_sizes = largest_cluster(
                positions[selected],
                box_size,
                cut_off=self.cluster_cut_off,
            )
            max_cluster_size = int(cluster_sizes.get(1, 0))
            largest = cluster_labels == 1
            selected_labels = labels[selected]

            for name in self.cluster:
                count = int(np.count_nonzero(largest & (selected_labels == name)))
                ratio = count / max_cluster_size if max_cluster_size else 0.0
                structure_counts[f"{name} ratio"] = f"{ratio:.2f}"

            if self.index_file is not None:
                atoms = np.sort(self.ids[selected][largest])
                self.index_file.write(
                    f"{simulation.currentStep}\t"
                    f"{' '.join(str(index) for index in atoms)}\n"
                )
                self.index_file.flush()
        else:
            max_cluster_size = 0
            for name in self.cluster:
                structure_counts[f"{name} ratio"] = "0.00"

            if self.index_file is not None:
                self.index_file.write(f"{simulation.currentStep}\t\n")
                self.index_file.flush()

        structure_counts["Max Cluster"] = max_cluster_size

        temperature = self._temperature(simulation)
        time = self._time(simulation)
        row = [
            str(simulation.currentStep),
            f"{time:.4f}",
            f"{temperature:.2f}",
        ]
        row.extend(str(structure_counts[name]) for name in self.structures)
        self.file.write("\t".join(row) + "\n")
        self.file.flush()

        self.max_cluster = max_cluster_size
        self.last_counts = structure_counts

    def close(self) -> None:
        """Flush and close the output files.  Safe to call more than once."""
        if getattr(self, "_closed", True):
            return
        file = getattr(self, "file", None)
        if file is not None and not file.closed:
            file.close()
        index_file = getattr(self, "index_file", None)
        if index_file is not None and not index_file.closed:
            index_file.close()
        self._closed = True

    def __enter__(self) -> ChillPlusReporter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self.close()


__all__ = ["ChillPlusReporter"]
