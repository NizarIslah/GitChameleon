# library: networkx
# version: 2.7
# extra_dependencies: ['numpy==1.21.2']
import networkx as nx


def modularity_communities(G: nx.Graph) -> list:
    return nx.community.greedy_modularity_communities(G, n_communities=5)
