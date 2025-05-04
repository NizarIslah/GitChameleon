# library: networkx
# version: 2.5
# extra_dependencies: []
import networkx as nx


def get_nodes(G: nx.Graph) -> list:
    return list(G.nodes)
