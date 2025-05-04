# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
from sample_32 import naive_modularity_communities


G = nx.karate_club_graph()
naive_modularity_communities_result = len(
    list(nx.community._naive_greedy_modularity_communities(G))
)
assert (
    len(list(naive_modularity_communities(G))) > 0
    and len(list(naive_modularity_communities(G)))
    == naive_modularity_communities_result
)
