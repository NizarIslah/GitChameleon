# library: networkx
# version: 2.6
# extra_dependencies: []
import networkx as nx


def bounding_distance(G: nx.Graph) -> int:
    return nx.algorithms.distance_measures.extrema_bounding(G, "diameter")
