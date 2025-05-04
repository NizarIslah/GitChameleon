# library: networkx
# version: 2.5
# extra_dependencies: []
import networkx as nx


def shortest_path(G: nx.Graph, source: int) -> list:
    return nx.bellman_ford_predecessor_and_distance(G, source)
