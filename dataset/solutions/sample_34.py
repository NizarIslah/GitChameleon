# library: networkx
# version: 2.5
# extra_dependencies: []
import networkx as nx


def get_first_edge(G: nx.Graph) -> tuple:
    return list(G.edges)[0]
