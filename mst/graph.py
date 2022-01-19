import numpy as np
import heapq
from typing import Union, Set


class Graph:
    def __init__(self, adjacency_mat: Union[np.ndarray, str], construct: bool = False):
        """Takes a adjacency matrix encoding a graph for construction of a minimum spanning tree using Prim's algorithm.

        Args:
            adjacency_mat (Union[np.ndarray, str]): Path to adjacency matrix to load with delimiter ',' or adjacency matrix in numpy array form.
            construct (bool, optional): Whether to create minimum spanning tree immediately. Defaults to False.

        Raises:
            TypeError: If adjacency matrix is not an np.ndarray or a valid path.
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else:
            raise TypeError("Input must be a valid path or an adjacency matrix")

        if self.adj_mat.dtype != float:
            self.adj_mat = self.adj_mat.astype(float)

        if np.allclose(self.adj_mat, self.adj_mat.T):
            raise ValueError("Adjacency matrix is asymmetric.")

        self.num_nodes = len(self.adj_mat)
        self.queue = []

        self.mst = None
        if construct:
            self.construct_mst()

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=",")

    def _add_connected_nodes_to_queue(
        self, node: int, unused_nodes: Set[int], adj_mat: np.ndarray
    ):
        """
        Will add neighbors of `node` that we haven't put in the tree already (not in unused_nodes set) into the queue.
        Queue is a list/heapq that holds elements representing vertices that are (weight, node @ start of vertex, node @ end of vertex)

        Args:
            node (int): node index
            unused_nodes (Set[int]): set of nodes we have already added to the tree
            adj_mat (np.ndarray): adjacency matrix we will build the tree from
        """

        for unused_node_idx in unused_nodes:
            weight = adj_mat[node, unused_node_idx]
            edge_tuple = (weight, node, unused_node_idx)
            heapq.heappush(self.queue, edge_tuple)

    def _get_next_edge(self, nodes: Set[int]):
        """
        Will use pop vertices (weight, start_node, end_node) where start_node and end_node \
            are node indices until we find one where end_node isn't in the nodes set.

        Args:
            nodes (Set[int]): A set of nodes that we haven't seen yet.

        Returns:
            tuple representing the next vertex in the tree: (weight of node, node @ start of vertex, node @ end of vertex)
        """

        end_node = None

        while (
            end_node not in nodes
        ):  # this will run until we get a vertex for which end_node is a new node we haven't seen yet
            weight, start_node, end_node = heapq.heappop(self.queue)

        return weight, start_node, end_node

    def construct_mst(self):
        """
        Will construct a minimum spanning tree (MST) from the given connected symmetric adjacency matrix.
        We:
            1. select a random vertex, find neighbors and put them into self.queue (_add_connected_nodes_to_queue)
            2. pop minimum weight vertex (indices u, v) from self.queue and if we haven't seen node v, add it to the tree
            3. continue until we've added every node

        Sets:
            self.mst: np array with lower triangular entries as connections in a minimum spanning tree.

        TODO: I would guess there's some way to make it work for a disconnected graph, but I'm not sure how to go about it..
        """

        adj_mat = self.adj_mat.copy()
        np.nan_to_num(adj_mat, nan=np.inf, copy=False)

        adj_mat[
            adj_mat == 0
        ] = np.inf  # prevent unconnected entries from being added to tree artificially

        self.mst = np.zeros_like(adj_mat)  # set of mst placeholder matrix

        nodes = list(
            range(self.num_nodes)
        )  # set up list of nodes; we will use this to track which nodes we used/saw already

        start_node = np.random.choice(nodes, size=1).item()  # get start node
        nodes = set(nodes)

        # adjust num_nodes and remove start_node from set of seen nodes
        num_nodes = self.num_nodes - 1
        nodes.remove(start_node)

        self._add_connected_nodes_to_queue(
            start_node, nodes, adj_mat
        )  # add vertices with start_node at the start to the queue

        while num_nodes > 0:  # keep going until we added all nodes
            weight, start_node, end_node = self._get_next_edge(nodes)

            # make sure we return a lower triangular matrix
            if start_node > end_node:  # so we can return an lower triangular matrix
                self.mst[start_node, end_node] = weight
            else:
                self.mst[end_node, start_node] = weight

            nodes.remove(end_node)  # remove end_node since we've used it already
            self._add_connected_nodes_to_queue(
                end_node, nodes, adj_mat
            )  # add neighbors of next node to queue
            num_nodes -= 1

        np.nan_to_num(
            self.mst, posinf=0, copy=False
        )  # if there are np.infs, it means there are n>1 connected components; we will zero these out
        # this is unnecesssary, I didn't realize before we didn't need to worry about unconnected graphs, but probably doesn't hurt
