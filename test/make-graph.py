import matplotlib.pyplot as plt
from networkx.generators import random_graphs
import networkx as nx
import numpy as np
import pathlib

folder = pathlib.Path(__file__).resolve().parent
graph = random_graphs.erdos_renyi_graph(10, p=0.25, seed=100, directed=False)
# graph.add_edge(3, 1)

adj_mat = nx.to_numpy_array(graph)
adj_mat[np.tril_indices_from(adj_mat)] = 0

nonzero = np.nonzero(adj_mat)
np.random.seed(0)
for (i, j) in zip(*nonzero):
    rand = np.random.randint(1, 10)
    adj_mat[i, j] = rand
    adj_mat[j, i] = rand

np.savetxt(folder / "erdosrenyi.txt", adj_mat)

# write graph picture
options = {"node_color": "white", "edgecolors": "black", "linewidths": 2}

# nx.draw_networkx(graph, **options)
# plt.savefig(folder / "erdosrenyi.png")
