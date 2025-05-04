# library: networkx
# version: 2.5
# extra_dependencies: []
import networkx as nx


def naive_modularity_communities(G: nx.Graph) -> list:
    return nx.community.naive_greedy_modularity_communities(G)
