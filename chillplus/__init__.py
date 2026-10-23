"""CHILL+ structure identification and an OpenMM reporter.

The package provides two layers:

``chillplus.core``
    The CHILL+ classifier for oxygen coordinates (NumPy/SciPy only).
``chillplus.reporter``
    :class:`ChillPlusReporter`, an OpenMM reporter built on the classifier.
"""

from .core import (
    CLATHRATE_STRUCTURES,
    DEFAULT_CLUSTER_CUTOFF,
    DEFAULT_CUTOFF,
    ICE_STRUCTURES,
    STRUCTURE_TYPES,
    cartesian_to_spherical,
    classify,
    classify_structure,
    correlation,
    find_nearest_neighbors,
    largest_cluster,
    spherical_harmonics,
)
from .reporter import ChillPlusReporter

__version__ = "1.0.0"

__all__ = [
    "CLATHRATE_STRUCTURES",
    "DEFAULT_CLUSTER_CUTOFF",
    "DEFAULT_CUTOFF",
    "ICE_STRUCTURES",
    "STRUCTURE_TYPES",
    "ChillPlusReporter",
    "__version__",
    "cartesian_to_spherical",
    "classify",
    "classify_structure",
    "correlation",
    "find_nearest_neighbors",
    "largest_cluster",
    "spherical_harmonics",
]
