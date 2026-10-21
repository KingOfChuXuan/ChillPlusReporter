"""CHILL+ structure identification for water, ice, and clathrate hydrates.

This module implements the CHILL+ algorithm of Nguyen and Molinero
[J. Phys. Chem. B 2015, 119, 9369-9376, doi:10.1021/jp510289t] for
classifying water oxygen atoms as cubic ice (Ic), hexagonal ice (Ih),
interfacial ice, clathrate hydrate, interfacial clathrate, or disordered
water ("Other").

Only the positions of the oxygen atoms are required.  The implementation
follows the reference algorithm and the original ``ChillPlus.py`` script by
Xuan Jiang.  Periodic boundary conditions are supported for orthorhombic
boxes.

The main entry points are :func:`classify` (also available under its
historical name :data:`ChillPlus`) and :func:`largest_cluster`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import KDTree
from scipy.special import sph_harm_y

#: All structure labels produced by :func:`classify`, in output order.
STRUCTURE_TYPES: tuple[str, ...] = (
    "Ic",
    "Ih",
    "Interfacial Ice",
    "Clathrate",
    "Interfacial Clathrate",
    "Other",
)

#: Labels that count as crystalline ice.
ICE_STRUCTURES: tuple[str, ...] = ("Ic", "Ih")

#: Labels that count as clathrate hydrate.
CLATHRATE_STRUCTURES: tuple[str, ...] = (
    "Clathrate",
    "Interfacial Clathrate",
)

#: Default O-O cutoff used to define the first coordination shell (angstrom).
DEFAULT_CUTOFF: float = 3.5

#: Default cutoff used to connect molecules into ice nuclei (angstrom).
DEFAULT_CLUSTER_CUTOFF: float = 3.2


def _validate_positions(
    positions: Sequence[Sequence[float]] | np.ndarray,
    box_size: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated ``(positions, box_size)`` arrays.

    Parameters
    ----------
    positions : array_like, shape (n, 3)
        Cartesian coordinates of the oxygen atoms in angstrom.
    box_size : array_like, shape (3,)
        Orthorhombic periodic box lengths in angstrom.

    Returns
    -------
    positions : ndarray, shape (n, 3)
        Coordinates as ``float64``.
    box_size : ndarray, shape (3,)
        Box lengths as ``float64``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box_size, dtype=np.float64)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (n, 3), got {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions contain NaN or infinite values")
    if box.shape != (3,):
        raise ValueError(f"box_size must have shape (3,), got {box.shape}")
    if not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_size entries must be finite and positive")

    return pos, box


def _validate_cutoff(cut_off: float) -> float:
    """Return ``cut_off`` as a positive float."""
    value = float(cut_off)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("cut_off must be a finite positive number")
    return value


def cartesian_to_spherical(
    cartesian_coords: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian vectors to polar and azimuthal angles.

    Parameters
    ----------
    cartesian_coords : array_like, shape (n, 3)
        Vectors to convert, typically displacement vectors in angstrom.

    Returns
    -------
    theta : ndarray, shape (n,)
        Polar angle in ``[0, pi]``.  The angle of a zero vector is set to 0.
    phi : ndarray, shape (n,)
        Azimuthal angle in ``[-pi, pi]``.
    """
    coords = np.asarray(cartesian_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"cartesian_coords must have shape (n, 3), got {coords.shape}")

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    radius = np.sqrt(x * x + y * y + z * z)
    cos_theta = np.divide(z, radius, out=np.zeros_like(radius), where=radius > 0.0)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta[radius == 0.0] = 0.0
    phi = np.arctan2(y, x)
    return theta, phi


def find_nearest_neighbors(
    positions: Sequence[Sequence[float]] | np.ndarray,
    box_size: Sequence[float] | np.ndarray,
    cut_off: float = DEFAULT_CUTOFF,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """Find all neighbours within ``cut_off`` using periodic boundaries.

    Parameters
    ----------
    positions : array_like, shape (n, 3)
        Oxygen coordinates in angstrom.  They may lie outside the box; they
        are wrapped into ``[0, box_size)`` internally.
    box_size : array_like, shape (3,)
        Orthorhombic periodic box lengths in angstrom.
    cut_off : float, optional
        Neighbour cutoff in angstrom.

    Returns
    -------
    neighbors : dict[int, ndarray]
        ``neighbors[k]`` is an integer array of shape ``(n_k, k)`` holding
        the indices of the ``k`` neighbours of each atom with exactly ``k``
        neighbours.
    neighbor_ids : dict[int, ndarray]
        ``neighbor_ids[k]`` holds the atom indices corresponding to the rows
        of ``neighbors[k]``.
    n_bad : int
        Number of atoms that do not have exactly four neighbours.  CHILL+
        treats those atoms as "Other".
    """
    pos, box = _validate_positions(positions, box_size)
    cutoff = _validate_cutoff(cut_off)

    wrapped = np.mod(pos, box)
    tree = KDTree(wrapped, boxsize=box)
    neighbor_lists = [
        np.fromiter((j for j in found if j != i), dtype=np.int64)
        for i, found in enumerate(tree.query_ball_point(wrapped, cutoff))
    ]

    counts = np.fromiter(
        (len(neighbors) for neighbors in neighbor_lists),
        dtype=np.int64,
        count=len(neighbor_lists),
    )

    neighbors: dict[int, np.ndarray] = {}
    neighbor_ids: dict[int, np.ndarray] = {}
    for count in np.unique(counts):
        mask = counts == count
        ids = np.flatnonzero(mask)
        neighbors[int(count)] = np.array(
            [neighbor_lists[i] for i in ids], dtype=np.int64
        )
        neighbor_ids[int(count)] = ids

    n_bad = int(np.count_nonzero(counts != 4))
    return neighbors, neighbor_ids, n_bad


def spherical_harmonics(
    ref_points: Sequence[Sequence[float]] | np.ndarray,
    neighbors: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    num_neighbor: int,
    box_size: Sequence[float] | np.ndarray,
    l: int = 3,
) -> np.ndarray:
    """Compute neighbour-averaged spherical harmonics for reference points.

    Parameters
    ----------
    ref_points : array_like, shape (n, 3)
        Reference atom coordinates in angstrom.
    neighbors : array_like, shape (n, num_neighbor, 3)
        Coordinates of the neighbours of each reference atom.
    num_neighbor : int
        Number of neighbours per reference atom.
    box_size : array_like, shape (3,)
        Orthorhombic periodic box lengths in angstrom.
    l : int, optional
        Spherical-harmonic degree.  CHILL+ uses ``l=3``.

    Returns
    -------
    ndarray, shape (n, 2 * l + 1)
        Complex, neighbour-averaged :math:`q_{lm}` values ordered by
        ``m = -l, ..., l``.
    """
    ref = np.asarray(ref_points, dtype=np.float64)
    nbr = np.asarray(neighbors, dtype=np.float64)
    box = np.asarray(box_size, dtype=np.float64)

    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError(f"ref_points must have shape (n, 3), got {ref.shape}")
    if not isinstance(num_neighbor, (int, np.integer)) or num_neighbor <= 0:
        raise ValueError("num_neighbor must be a positive integer")
    if nbr.shape != (ref.shape[0], int(num_neighbor), 3):
        raise ValueError(
            "neighbors must have shape "
            f"({ref.shape[0]}, {num_neighbor}, 3), got {nbr.shape}"
        )
    if box.shape != (3,):
        raise ValueError(f"box_size must have shape (3,), got {box.shape}")
    if not isinstance(l, (int, np.integer)) or l < 0:
        raise ValueError("l must be a non-negative integer")

    n = ref.shape[0]
    order = 2 * int(l) + 1

    # Minimum-image displacement from each reference atom to its neighbours.
    displacement = nbr - ref[:, np.newaxis, :]
    displacement -= box * np.round(displacement / box)

    theta, phi = cartesian_to_spherical(displacement.reshape(-1, 3))
    theta = theta.reshape(n, int(num_neighbor), 1)
    phi = phi.reshape(n, int(num_neighbor), 1)
    m_values = np.arange(-int(l), int(l) + 1).reshape(1, 1, order)

    harmonics = sph_harm_y(int(l), m_values, theta, phi)
    return np.mean(harmonics, axis=1)


def correlation(qlm1: np.ndarray, qlm2: np.ndarray) -> np.ndarray:
    """Compute normalized real correlation coefficients between ``qlm`` sets.

    Parameters
    ----------
    qlm1 : array_like, shape (n, 2 * l + 1)
        Reference ``q_lm`` vectors.
    qlm2 : array_like, shape (n, k, 2 * l + 1)
        Candidate neighbour ``q_lm`` vectors.  ``k`` is normally 4.

    Returns
    -------
    ndarray, shape (n, k)
        Real correlation values in ``[-1, 1]``.
    """
    first = np.asarray(qlm1, dtype=np.complex128)
    second = np.asarray(qlm2, dtype=np.complex128)

    if first.ndim != 2:
        raise ValueError(f"qlm1 must be two-dimensional, got {first.shape}")
    if second.ndim != 3:
        raise ValueError(f"qlm2 must be three-dimensional, got {second.shape}")
    if first.shape[0] != second.shape[0] or first.shape[1] != second.shape[2]:
        raise ValueError(
            f"qlm1 and qlm2 have incompatible shapes: {first.shape} and {second.shape}"
        )
    if first.shape[0] == 0:
        return np.empty((0, second.shape[1]), dtype=np.float64)

    norm1 = np.linalg.norm(first, axis=1, keepdims=True)
    norm1[norm1 == 0.0] = 1.0
    norm2 = np.linalg.norm(second, axis=2, keepdims=True)
    norm2[norm2 == 0.0] = 1.0

    values = np.sum(
        (first / norm1)[:, np.newaxis, :] * np.conjugate(second / norm2),
        axis=2,
    )
    if np.max(np.abs(values.imag)) < 1e-10:
        return np.real(values)
    raise ValueError(
        "correlation coefficients have a non-negligible imaginary part; "
        "the q_lm vectors are inconsistent"
    )


def classify_structure(c_values: np.ndarray) -> np.ndarray:
    """Classify atoms from their four CHILL+ correlation coefficients.

    The classification uses the number of staggered bonds ``S``
    (``C <= -0.8``) and eclipsed bonds ``E`` (``-0.35 <= C <= 0.25``)
    exactly as defined by Nguyen and Molinero.

    Parameters
    ----------
    c_values : array_like, shape (n, 4)
        Correlation coefficients between an atom and its four neighbours.

    Returns
    -------
    ndarray of str, shape (n,)
        One of :data:`STRUCTURE_TYPES` for each atom.
    """
    values = np.asarray(c_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"c_values must have shape (n, 4), got {values.shape}")

    eclipsed = np.count_nonzero((values >= -0.35) & (values <= 0.25), axis=1)
    staggered = np.count_nonzero(values <= -0.8, axis=1)

    categories = np.full(values.shape[0], "Other", dtype=object)
    categories[(eclipsed == 0) & (staggered == 4)] = "Ic"
    categories[(eclipsed == 1) & (staggered == 3)] = "Ih"
    categories[(staggered == 2) | ((staggered == 3) & (eclipsed == 0))] = (
        "Interfacial Ice"
    )
    categories[(eclipsed == 4) & (staggered == 0)] = "Clathrate"
    categories[eclipsed == 3] = "Interfacial Clathrate"
    return categories


def classify(
    positions: Sequence[Sequence[float]] | np.ndarray,
    box_size: Sequence[float] | np.ndarray,
    cut_off: float = DEFAULT_CUTOFF,
    l: int = 3,
) -> tuple[dict[str, int], np.ndarray]:
    """Classify water oxygen atoms with the CHILL+ algorithm.

    Parameters
    ----------
    positions : array_like, shape (n, 3)
        Oxygen coordinates in angstrom.  Coordinates outside the box are
        wrapped into the periodic box.
    box_size : array_like, shape (3,)
        Orthorhombic periodic box lengths in angstrom.
    cut_off : float, optional
        O-O cutoff defining the first coordination shell, in angstrom.
        Default is 3.5.
    l : int, optional
        Spherical-harmonic degree.  Default is 3, as in CHILL+.

    Returns
    -------
    counts : dict[str, int]
        Number of atoms in each of :data:`STRUCTURE_TYPES`.
    labels : ndarray of str, shape (n,)
        Structure label of every input atom.
    """
    pos, box = _validate_positions(positions, box_size)
    cutoff = _validate_cutoff(cut_off)
    if not isinstance(l, (int, np.integer)) or l < 0:
        raise ValueError("l must be a non-negative integer")
    degree = int(l)

    wrapped = np.mod(pos, box)
    n = wrapped.shape[0]
    labels = np.full(n, "", dtype=object)

    neighbors, neighbor_ids, _ = find_nearest_neighbors(wrapped, box, cutoff)

    q_lm = np.zeros((n, 2 * degree + 1), dtype=np.complex128)
    for num_neighbors, indices in neighbors.items():
        if num_neighbors == 0:
            continue
        ref_ids = neighbor_ids[num_neighbors]
        neighbor_positions = wrapped[indices]
        q_lm[ref_ids] = spherical_harmonics(
            wrapped[ref_ids],
            neighbor_positions,
            num_neighbors,
            box,
            degree,
        )

    if 4 in neighbors:
        good_ids = neighbor_ids[4]
        good_neighbors = neighbors[4]
        coefficients = correlation(q_lm[good_ids], q_lm[good_neighbors])
        labels[good_ids] = classify_structure(coefficients)

    # Atoms without exactly four neighbours are not classified above.
    labels[labels == ""] = "Other"

    counts = {name: int(np.count_nonzero(labels == name)) for name in STRUCTURE_TYPES}
    return counts, labels


def largest_cluster(
    positions: Sequence[Sequence[float]] | np.ndarray,
    box_size: Sequence[float] | np.ndarray,
    cut_off: float = DEFAULT_CLUSTER_CUTOFF,
) -> tuple[np.ndarray, dict[int, int]]:
    """Group atoms into periodic clusters using a distance cutoff.

    This is typically applied to the subset of atoms classified as ice.  The
    largest cluster (label ``1``) is the largest connected ice nucleus.

    Parameters
    ----------
    positions : array_like, shape (n, 3)
        Coordinates in angstrom.  Coordinates outside the box are wrapped.
    box_size : array_like, shape (3,)
        Orthorhombic periodic box lengths in angstrom.
    cut_off : float, optional
        Distance cutoff in angstrom.  Default is 3.2.

    Returns
    -------
    labels : ndarray of int, shape (n,)
        Cluster label of every atom.  Label ``1`` is the largest cluster,
        label ``2`` the second largest, and so on.  Unclustered atoms cannot
        occur because every atom belongs to some cluster.
    sizes : dict[int, int]
        Mapping from cluster label to cluster size, ordered by decreasing
        size.
    """
    pos, box = _validate_positions(positions, box_size)
    cutoff = _validate_cutoff(cut_off)
    n = pos.shape[0]
    labels = np.zeros(n, dtype=np.int64)
    if n == 0:
        return labels, {}

    wrapped = np.mod(pos, box)
    tree = KDTree(wrapped, boxsize=box)
    visited = np.zeros(n, dtype=bool)
    clusters: list[np.ndarray] = []

    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        members = [start]
        frontier = np.array([start], dtype=np.int64)
        while frontier.size:
            found = tree.query_ball_point(wrapped[frontier], cutoff)
            candidates = np.concatenate(
                [np.asarray(indices, dtype=np.int64) for indices in found]
            )
            candidates = np.unique(candidates)
            candidates = candidates[~visited[candidates]]
            visited[candidates] = True
            members.extend(candidates.tolist())
            frontier = candidates
        clusters.append(np.asarray(members, dtype=np.int64))

    clusters.sort(key=len, reverse=True)
    sizes: dict[int, int] = {}
    for cluster_label, members in enumerate(clusters, start=1):
        labels[members] = cluster_label
        sizes[cluster_label] = len(members)
    return labels, sizes


#: Historical name kept for scripts written against the original module.
ChillPlus = classify

#: Historical name kept for scripts written against the original module.
KDTreeCluster = largest_cluster

__all__ = [
    "CLATHRATE_STRUCTURES",
    "DEFAULT_CLUSTER_CUTOFF",
    "DEFAULT_CUTOFF",
    "ICE_STRUCTURES",
    "STRUCTURE_TYPES",
    "ChillPlus",
    "KDTreeCluster",
    "cartesian_to_spherical",
    "classify",
    "classify_structure",
    "correlation",
    "find_nearest_neighbors",
    "largest_cluster",
    "spherical_harmonics",
]
