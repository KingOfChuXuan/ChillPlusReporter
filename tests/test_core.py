"""Unit tests for the CHILL+ core algorithm."""

import numpy as np
import pytest

from chillplus import (
    STRUCTURE_TYPES,
    cartesian_to_spherical,
    classify,
    classify_structure,
    correlation,
    find_nearest_neighbors,
    largest_cluster,
    spherical_harmonics,
)


def test_structure_types():
    assert STRUCTURE_TYPES == (
        "Ic",
        "Ih",
        "Interfacial Ice",
        "Clathrate",
        "Interfacial Clathrate",
        "Other",
    )


def test_cartesian_to_spherical():
    theta, phi = cartesian_to_spherical(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    np.testing.assert_allclose(theta, [np.pi / 2, np.pi / 2, 0.0])
    np.testing.assert_allclose(phi, [0.0, np.pi / 2, 0.0])

    theta, phi = cartesian_to_spherical(np.zeros((2, 3)))
    np.testing.assert_array_equal(theta, [0.0, 0.0])
    np.testing.assert_array_equal(phi, [0.0, 0.0])


def test_cubic_ice_is_ic(cubic_ice):
    positions, box_size = cubic_ice
    counts, labels = classify(positions, box_size)

    assert counts == {
        "Ic": 64,
        "Ih": 0,
        "Interfacial Ice": 0,
        "Clathrate": 0,
        "Interfacial Clathrate": 0,
        "Other": 0,
    }
    np.testing.assert_array_equal(labels, ["Ic"] * 64)


def test_positions_outside_box_are_wrapped(cubic_ice):
    positions, box_size = cubic_ice
    shifted = positions - 3.0 * box_size
    counts, labels = classify(shifted, box_size)
    assert counts["Ic"] == 64
    np.testing.assert_array_equal(labels, ["Ic"] * 64)


def test_wrap_rounding_at_box_boundary():
    box_size = np.array([50.441, 50.441, 50.441])
    positions = np.array([[-1e-16, 1.0, 1.0], [1.5, 1.0, 1.0]])

    # np.mod(-1e-16, 50.441) rounds to 50.441, which KDTree rejects.
    neighbors, neighbor_ids, n_bad = find_nearest_neighbors(
        positions, box_size, cut_off=3.5
    )
    assert n_bad == 2  # both atoms have one neighbour, not four
    assert set(neighbors) == {1}
    assert neighbor_ids[1].shape == (2,)

    labels, sizes = largest_cluster(positions, box_size, cut_off=3.2)
    assert sizes == {1: 2}
    np.testing.assert_array_equal(labels, [1, 1])


def test_neighbor_search_uses_periodic_boundaries(cubic_ice):
    positions, box_size = cubic_ice
    neighbors, neighbor_ids, n_bad = find_nearest_neighbors(
        positions, box_size, cut_off=3.5
    )

    assert n_bad == 0
    assert set(neighbors) == {4}
    assert neighbors[4].shape == (64, 4)
    assert neighbor_ids[4].shape == (64,)
    np.testing.assert_array_equal(neighbor_ids[4], np.arange(64))
    assert np.all(neighbors[4] != np.arange(64)[:, None])


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ([-0.9, -0.9, -0.9, -0.9], "Ic"),
        ([-0.9, -0.9, -0.9, 0.0], "Ih"),
        ([-0.9, -0.9, 0.0, 0.0], "Interfacial Ice"),
        ([-0.9, -0.9, -0.9, 0.5], "Interfacial Ice"),
        ([0.0, 0.0, 0.0, 0.0], "Clathrate"),
        ([-0.9, 0.0, 0.0, 0.0], "Interfacial Clathrate"),
        ([0.5, 0.5, 0.5, 0.5], "Other"),
    ],
)
def test_classify_structure_thresholds(coefficients, expected):
    categories = classify_structure(np.asarray([coefficients], dtype=float))
    assert categories[0] == expected


def test_classify_structure_accepts_threshold_boundaries():
    categories = classify_structure(
        np.array(
            [
                [-0.8, -0.35, -1.0, -1.0],  # S=3, E=1
                [-0.9, -0.9, -0.9, 0.25],  # upper E boundary; S=3, E=1
                [0.25, 0.5, 0.5, 0.5],  # E=1, S=0
            ],
            dtype=float,
        )
    )
    np.testing.assert_array_equal(categories, ["Ih", "Ih", "Other"])


def test_correlation_is_normalized_and_real():
    rng = np.random.default_rng(42)
    q_lm = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))
    neighbors = np.repeat(q_lm[:, np.newaxis, :], 4, axis=1)

    values = correlation(q_lm, neighbors)
    assert values.shape == (5, 4)
    assert np.isrealobj(values)
    np.testing.assert_allclose(values, 1.0)


def test_correlation_handles_zero_vectors():
    q_lm = np.zeros((2, 7), dtype=complex)
    neighbors = np.zeros((2, 4, 7), dtype=complex)
    values = correlation(q_lm, neighbors)
    np.testing.assert_array_equal(values, np.zeros((2, 4)))


def test_correlation_handles_zero_candidates():
    values = correlation(np.ones((2, 7)), np.ones((2, 0, 7)))
    assert values.shape == (2, 0)
    assert values.dtype == np.float64


def test_largest_cluster_orders_by_size():
    first = np.array([[1.0 + 2.7 * i, 1.0, 1.0] for i in range(5)])
    second = np.array([[30.0 + 2.7 * i, 30.0, 30.0] for i in range(3)])
    positions = np.vstack([first, second])
    box_size = np.array([50.0, 50.0, 50.0])

    labels, sizes = largest_cluster(positions, box_size, cut_off=3.2)
    assert sizes == {1: 5, 2: 3}
    np.testing.assert_array_equal(labels, [1] * 5 + [2] * 3)


def test_spherical_harmonics_rejects_invalid_box():
    with pytest.raises(ValueError, match="finite and positive"):
        spherical_harmonics(
            np.zeros((1, 3)),
            np.zeros((1, 4, 3)),
            4,
            np.array([0.0, 10.0, 10.0]),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        spherical_harmonics(
            np.zeros((1, 3)),
            np.zeros((1, 4, 3)),
            4,
            np.array([10.0, np.nan, 10.0]),
        )


def test_empty_input():
    counts, labels = classify(np.empty((0, 3)), np.array([10.0, 10.0, 10.0]))
    assert counts == {name: 0 for name in STRUCTURE_TYPES}
    assert labels.shape == (0,)

    cluster_labels, sizes = largest_cluster(
        np.empty((0, 3)), np.array([10.0, 10.0, 10.0])
    )
    assert cluster_labels.shape == (0,)
    assert sizes == {}


def test_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        classify(np.zeros((4, 2)), np.array([10.0, 10.0, 10.0]))
    with pytest.raises(ValueError, match="box_size"):
        classify(np.zeros((4, 3)), np.array([10.0, -1.0, 10.0]))
    with pytest.raises(ValueError, match="cut_off"):
        classify(np.zeros((4, 3)), np.array([10.0, 10.0, 10.0]), cut_off=0)
    with pytest.raises(ValueError, match="non-negative"):
        classify(np.zeros((4, 3)), np.array([10.0, 10.0, 10.0]), l=-1)
    with pytest.raises(ValueError, match="shape"):
        classify_structure(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        correlation(np.zeros((2, 3)), np.zeros((2, 4, 2)))
