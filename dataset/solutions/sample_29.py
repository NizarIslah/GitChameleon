# library: networkx
# version: 2.8
# extra_dependencies: []
import networkx as nx


def bounding_distance(G: nx.Graph) -> int:
    return nx.diameter(G, usebounds=True)
