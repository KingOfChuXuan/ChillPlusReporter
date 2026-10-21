# ChillPlusReporter

[![tests](https://github.com/KingOfChuXuan/ChillPlusReporter/actions/workflows/tests.yml/badge.svg)](https://github.com/KingOfChuXuan/ChillPlusReporter/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CHILL+ structure identification for water, ice, and clathrate hydrates, with
an [OpenMM](https://openmm.org/) reporter.

The package classifies each water oxygen atom as cubic ice (**Ic**),
hexagonal ice (**Ih**), interfacial ice, clathrate hydrate, interfacial
clathrate, or disordered water (**Other**) using the staggered/eclipsed
water-water bond criterion of Nguyen and Molinero.  It also finds the largest
connected ice nucleus and writes its size, composition, and (optionally) its
oxygen indices during a simulation.

The classifier itself uses only NumPy and SciPy; OpenMM is required for the
reporter and is installed as a package dependency.  The reporter is an ordinary
OpenMM reporter and adds little overhead to the MD loop.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reporter options](#reporter-options)
- [Output format](#output-format)
- [Core API](#core-api)
- [Examples](#examples)
- [Performance](#performance)
- [Notes and limitations](#notes-and-limitations)
- [Migrating from the original scripts](#migrating-from-the-original-scripts)
- [Repository layout](#repository-layout)
- [References](#references)
- [Citation](#citation)
- [License](#license)

## Features

- CHILL+ classification of oxygen atoms into six structure types.
- Periodic boundary conditions for orthorhombic boxes, with a SciPy KD-tree
  neighbour search.
- Largest ice nucleus by periodic connected-component search.
- OpenMM reporter with configurable report interval, water residue name,
  analysed atom subset, cluster definition, and time unit.
- Optional output of the oxygen atom indices in the largest nucleus.
- Self-contained tests, including a 4096-water regression configuration.
- No compiled extensions; NumPy, SciPy, and OpenMM are the only runtime
  dependencies.

## Requirements

- Python 3.10 or newer
- NumPy 1.23 or newer
- SciPy 1.15 or newer (`scipy.special.sph_harm_y`)
- OpenMM 8.2 or newer (tested with 8.5.2)

A ready-to-use conda environment is provided in `environment.yml`.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/KingOfChuXuan/ChillPlusReporter.git
cd ChillPlusReporter
pip install -e .
```

Alternatively, create the conda environment first:

```bash
conda env create -f environment.yml
conda activate chillplus
pip install -e .
```

For tests and development tools:

```bash
pip install -e ".[test]"
python -m pytest
```

## Quick start

```python
from openmm.app import Simulation
from chillplus import ChillPlusReporter

simulation = Simulation(topology, system, integrator)

reporter = ChillPlusReporter("structures.dat", 5000)
simulation.reporters.append(reporter)

simulation.step(100000)
reporter.close()
```

The reporter analyses all oxygen atoms of residues named `HOH` by default and
writes one row every 5000 steps.  `reporter.max_cluster` holds the largest ice
nucleus size of the most recent frame, which is convenient for stopping a
simulation once freezing or melting is complete.

## Reporter options

```python
ChillPlusReporter(
    file,
    interval,
    need_index=False,
    group=None,
    cut_off=3.5,
    l=3,
    cluster=None,
    cluster_cut_off=3.2,
    water_name="HOH",
    time_unit=nanosecond,
)
```

| Argument | Meaning | Default |
| --- | --- | --- |
| `file` | Output file.  `.dat` is appended when the path has no suffix. | required |
| `interval` | Report interval in simulation steps. | required |
| `need_index` | Also write oxygen indices of the largest nucleus. | `False` |
| `group` | Atom indices to analyse instead of all matching oxygens. | all O atoms |
| `cut_off` | O-O cutoff for the first coordination shell, in angstrom. | `3.5` |
| `l` | Spherical-harmonic degree. | `3` |
| `cluster` | Structure labels that count as the nucleus. | `("Ic", "Ih")` |
| `cluster_cut_off` | Distance cutoff used to connect atoms into nuclei, in angstrom. | `3.2` |
| `water_name` | Residue name of water in the topology. | `"HOH"` |
| `time_unit` | OpenMM unit used in the time column. | `nanosecond` |

### Different water residue name

```python
reporter = ChillPlusReporter("structures.dat", 5000, water_name="WAT")
```

### Only part of the system

```python
oxygen_indices = [
    atom.index
    for atom in topology.atoms()
    if atom.residue.name == "HOH"
    and atom.element is not None
    and atom.element.symbol == "O"
]
reporter = ChillPlusReporter("structures.dat", 5000, group=oxygen_indices)
```

The analysed and unanalysed water molecules must not be in contact.  The
neighbour count of an analysed atom is computed only from the selected subset,
so cutting through a hydrogen-bonded network changes the classification at the
boundary.

### Largest nucleus indices

```python
reporter = ChillPlusReporter("structures.dat", 5000, need_index=True)
```

This writes `<stem>_index<suffix>` next to the main output file.  The indices
are zero-based atom indices from the OpenMM topology.

### Stop when freezing is complete

```python
reporter = ChillPlusReporter("structures.dat", 5000)
simulation.reporters.append(reporter)

for _ in range(100):
    simulation.step(5000)
    if reporter.max_cluster is not None and reporter.max_cluster > 1000:
        break
```

## Output format

The main output is tab-separated and can be read with NumPy, pandas, or a
spreadsheet.  A default header looks like:

```text
#Step  Time(ns)  Temperature(K)  Ic  Ih  Max Cluster  Ic ratio  Ih ratio  Interfacial Ice  Clathrate  Interfacial Clathrate  Other
```

| Column | Meaning |
| --- | --- |
| `#Step` | Simulation step. |
| `Time(ns)` | Simulation time in `time_unit` (the label follows the unit). |
| `Temperature(K)` | Integrator temperature, or a context parameter whose name contains `Temperature`; `nan` if unavailable. |
| `Ic`, `Ih`, ... | Number of atoms of each structure type. |
| `Max Cluster` | Size of the largest connected nucleus formed by the `cluster` labels. |
| `Ic ratio`, `Ih ratio` | Fraction of the largest nucleus with each cluster label. |
| Remaining columns | Structure types not included in `cluster`. |

The column order after `Temperature(K)` is: the labels in `cluster`, then
`Max Cluster`, then one ratio column per cluster label, then the remaining
structure types in their standard order.

The optional index file starts with `#Step\tOxygen index` and contains one row
per report frame:

```text
5000    12 45 78 91 ...
```

The second field is a space-separated list of zero-based oxygen atom indices.
An empty nucleus is written as an empty second field.

## Core API

The classifier works on oxygen coordinates in angstrom and an orthorhombic
periodic box:

```python
import numpy as np
from chillplus import classify, largest_cluster

positions = np.asarray(positions)  # shape (n, 3), angstrom
box_size = np.asarray(box_size)  # shape (3,), angstrom

counts, labels = classify(positions, box_size)
# counts: {'Ic': ..., 'Ih': ..., 'Interfacial Ice': ..., ...}
# labels: array of structure names, shape (n,)

ice = np.isin(labels, ("Ic", "Ih"))
cluster_labels, cluster_sizes = largest_cluster(positions[ice], box_size, cut_off=3.2)
print(cluster_sizes[1])  # largest ice nucleus
```

Coordinates outside the box are wrapped automatically.  The historical names
`ChillPlus` and `KDTreeCluster` are kept as aliases for `classify` and
`largest_cluster`.

The lower-level functions are also available:

| Function | Purpose |
| --- | --- |
| `find_nearest_neighbors` | Periodic neighbour search grouped by coordination number. |
| `spherical_harmonics` | Neighbour-averaged `q_lm` values. |
| `correlation` | Normalized real correlation coefficients. |
| `classify_structure` | Apply the CHILL+ staggered/eclipsed thresholds. |
| `cartesian_to_spherical` | Polar and azimuthal angles of displacement vectors. |

## Examples

The bundled `examples/data/195K.pdb` contains 4096 water molecules.  Classify
it with:

```bash
python examples/analyze_pdb.py
```

Expected structure counts:

```text
Ic                       1109
Ih                        544
Interfacial Ice          1047
Clathrate                   5
Interfacial Clathrate      62
Other                    1329
```

The largest ice nucleus contains 1561 oxygen atoms, in agreement with the
reference implementation.

A minimal OpenMM simulation that demonstrates the reporter is provided in
`examples/reporter_demo.py`:

```bash
python examples/reporter_demo.py --output demo.dat --index
```

The demo uses an ideal cubic-ice lattice of oxygen atoms, not a physical water
model; it only shows how to attach the reporter and how the output is
formatted.

## Performance

The core classifier is vectorized and uses a SciPy KD-tree.  On a typical
laptop CPU, classifying 4096 oxygen atoms (the bundled 195 K configuration)
takes a few tens of milliseconds.  Runtime scales with the number of atoms and
the local density; the neighbour search dominates.  The reporter adds this
cost only on report frames, not on every MD step.

## Notes and limitations

- Pass **oxygen positions only**, one row per water molecule.  Hydrogens are
  not used by CHILL+.
- Coordinates and box lengths are in angstrom in the Python API.  The OpenMM
  reporter converts from OpenMM units internally.
- Only orthorhombic periodic boxes are supported.  The reporter raises
  `NotImplementedError` for triclinic boxes, and the core API assumes the three
  box lengths are independent.
- For dense systems, the neighbour cutoff should be smaller than half the
  shortest box length so that the minimum-image convention is unambiguous.
- The default CHILL+ parameters are `cut_off=3.5` angstrom, `l=3`, and
  `cluster_cut_off=3.2` angstrom.  They are exposed for flexibility but should
  normally not be changed when comparing with published CHILL+ results.
- The reporter writes the temperature from the integrator when available.  For
  integrators without `getTemperature()`, it looks for a context parameter
  whose name contains `Temperature`; otherwise the column is `nan`.
- The output file is flushed after every report frame, so it can be inspected
  while a simulation is running.

## Migrating from the original scripts

The original files used top-level module names.  The package now exposes the
same functionality through `chillplus`:

| Original | New |
| --- | --- |
| `from ChillPlus import ChillPlus` | `from chillplus import classify` |
| `from ChillPlus import KDTreeCluster` | `from chillplus import largest_cluster` |
| `from ChillPlusReporter import ChillPlusReporter` | `from chillplus import ChillPlusReporter` |

The main classifier and reporter signatures are unchanged.  The index file now
uses a plain tab-separated `step<TAB>indices` layout instead of the old
`step:<TAB>indices` layout.

Install the package first (`pip install -e .`) so that the examples and your
scripts can import `chillplus` from any working directory.

## Repository layout

```text
ChillPlusReporter/
├── chillplus/
│   ├── __init__.py       # public API
│   ├── core.py           # CHILL+ classifier and cluster search
│   ├── reporter.py       # OpenMM reporter
│   └── py.typed
├── examples/
│   ├── analyze_pdb.py
│   ├── reporter_demo.py
│   └── data/195K.pdb
├── tests/
│   ├── test_core.py
│   ├── test_example_data.py
│   └── test_reporter.py
├── .github/workflows/    # CI and release workflows
├── CITATION.cff
├── LICENSE
├── README.md
├── environment.yml
└── pyproject.toml
```

## References

The CHILL+ algorithm:

> Nguyen, A. H.; Molinero, V. *Identification of Clathrate Hydrates,
> Hexagonal Ice, Cubic Ice, and Liquid Water in Simulations: the CHILL+
> Algorithm.* J. Phys. Chem. B **2015**, 119 (29), 9369-9376.
> [doi:10.1021/jp510289t](https://doi.org/10.1021/jp510289t)

The original CHILL algorithm:

> Moore, E. B.; de la Llave, E.; Welke, K.; Scherlis, D. A.; Molinero, V.
> *Freezing, melting and structure of ice in a hydrophilic nanopore.*
> Phys. Chem. Chem. Phys. **2010**, 12, 4124-4134.
> [doi:10.1039/b919724a](https://doi.org/10.1039/b919724a)

OpenMM:

> Eastman, P.; et al. *OpenMM 7: Rapid development of high performance
> algorithms for molecular dynamics.* PLOS Comput. Biol. **2017**, 13,
> e1005659. [doi:10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659)

## Citation

If you use this package, please cite the CHILL+ paper above and this
repository.  A machine-readable citation is provided in
[`CITATION.cff`](CITATION.cff).

```bibtex
@software{Jiang2026ChillPlusReporter,
  author  = {Jiang, Xuan},
  title   = {ChillPlusReporter: CHILL+ structure analysis for OpenMM},
  year    = {2026},
  url     = {https://github.com/KingOfChuXuan/ChillPlusReporter},
  license = {MIT}
}
```

## License

This project is distributed under the [MIT License](LICENSE).
