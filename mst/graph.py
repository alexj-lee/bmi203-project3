from tkinter import W
import numpy as np
import heapq
from typing import Union, Set


class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """Unlike project 2, this Graph class takes an adjacency matrix as input. `adjacency_mat`
        can either be a 2D numpy array of floats or the path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else:
            raise TypeError("Input must be a valid path or an adjacency matrix")

        self.num_nodes = len(self.adj_mat)
        self.queue = []

        self.mst = None
        self.construct_mst()

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=",")

    def _add_connected_nodes_to_queue(self, node: int, unused_nodes: Set[int]):
        for unused_node_idx in unused_nodes:
            weight = self.adj_mat[node, unused_node_idx]
            edge_tuple = (weight, node, unused_node_idx)
            heapq.heappush(self.queue, edge_tuple)
        # print("queue state", self.queue)

    def _get_next_edge(self, nodes: Set[int]):
        end_node = None

        while end_node not in nodes:
            weight, start_node, end_node = heapq.heappop(self.queue)
        return weight, start_node, end_node

    def construct_mst(self):
        """Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`.

        `self.adj_mat` is a 2D numpy array of floats.
        Note that because we assume our input graph is undirected, `self.adj_mat` is symmetric.
        Row i and column j represents the edge weight between vertex i and vertex j. An edge weight of zero indicates that no edge exists.

        TODO:
            This function does not return anything. Instead, store the adjacency matrix
        representation of the minimum spanning tree of `self.adj_mat` in `self.mst`.
        We highly encourage the use of priority queues in your implementation. See the heapq
        module, particularly the `heapify`, `heappop`, and `heappush` functions.
        """

        np.nan_to_num(self.adj_mat, nan=np.inf, copy=False)
        self.adj_mat[self.adj_mat == 0] = np.inf
        self.mst = np.zeros_like(self.adj_mat)

        nodes = list(range(self.num_nodes))
        start_node = np.random.choice(nodes, size=1).item()
        nodes = set(nodes)
        self.num_nodes = self.num_nodes - 1
        nodes.remove(start_node)

        self._add_connected_nodes_to_queue(start_node, nodes)

        while self.num_nodes != 0:
            weight, start_node, end_node = self._get_next_edge(nodes)
            if start_node < end_node:  # so we can return an upper triangular matrix
                self.mst[start_node, end_node] = weight
            else:
                self.mst[end_node, start_node] = weight

            nodes.remove(end_node)
            self._add_connected_nodes_to_queue(end_node, nodes)
            self.num_nodes = self.num_nodes - 1
