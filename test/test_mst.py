# write tests for bfs
import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(
    adj_mat: np.ndarray,
    mst: np.ndarray,
    expected_weight: int,
    allowed_error: float = 0.0001,
):
    """Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot
    simply check for equality against a known MST of a graph.

    Arguments:
        adj_mat: Adjacency matrix of full graph
        mst: Adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: Allowed difference between proposed MST weight and `expected_weight`

    TODO:
        Add additional assertions to ensure the correctness of your MST implementation
    For example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?
    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    def correct_num_edges(mst):
        num_edges_found = len(np.nonzero(mst)[0])
        num_edges_expected = mst.shape[0] - 1
        return (
            num_edges_expected >= num_edges_found
        )  # spanning tree must have less than n-1 edges (fewer if graph is not connected)

    def correct_value_ranges(mst, adj_mat):
        is_correct = True
        mst_weight_indices = np.nonzero(mst)[0]

        min_vals = adj_mat.min(1)
        max_vals = adj_mat.max(1)

        for idx, mst_weight in enumerate(mst_weight_indices):
            if not (max_vals[idx] > mst_weight > min_vals[idx]):
                is_correct = True

        return is_correct

    def no_diag_entries(mst):
        return np.isclose(np.diag(mst).sum(), 0)

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i + 1):
            total += mst[i, j]

    assert approx_equal(
        total, expected_weight
    ), "Proposed MST has incorrect expected weight."

    assert correct_num_edges(mst), "Proposed MST has incorrect number of edges."
    assert correct_value_ranges(
        mst, adj_mat
    ), "Proposed MST has weights u, v that are not in the right range for corresponding adjacency matrix"
    assert no_diag_entries(
        mst
    ), "Proposed MST has self-links between nodes, which is not allowed."


def test_mst_small():
    """Unit test for the construction of a minimum spanning tree on a small graph"""
    file_path = "./data/small.csv"
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """Unit test for the construction of a minimum spanning tree using
    single cell data, taken from the Slingshot R package
    (https://bioconductor.org/packages/release/bioc/html/slingshot.html)
    """
    file_path = "./data/slingshot_example.txt"
    # load coordinates of single cells in low-dimensional subspace
    coords = np.loadtxt(file_path)
    # compute pairwise distances for all 140 cells to form an undirected weighted graph
    dist_mat = pairwise_distances(coords)
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    # see make-graph.py for steps to make graph; erdosrenyi.png for a graphic (doesn't include edge weights)

    file_path = "./test/erdosrenyi.txt"

    g = Graph(np.genfromtxt(file_path))
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 46.0)
