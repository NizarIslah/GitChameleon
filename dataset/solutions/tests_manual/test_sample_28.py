# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import numpy as np
import sample_28


G = nx.karate_club_graph()
result = nx.community.greedy_modularity_communities(G, n_communities=5)
assert len(modularity_communities(G)) > 0 and len(modularity_communities(G)) == len(
    result
)
