# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import sample_35


G = nx.path_graph(5)
shortest_path_result = nx.bellman_ford_predecessor_and_distance(G, 0)
assert shortest_path(G, 0) is not None and len(shortest_path(G, 0)) == len(
    shortest_path_result
)
