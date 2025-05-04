# library: networkx
# version: 2.8
# extra_dependencies: []
import networkx as nx


def modularity_communities(G: nx.Graph) -> list:
    return nx.community.greedy_modularity_communities(G, cutoff=5)
