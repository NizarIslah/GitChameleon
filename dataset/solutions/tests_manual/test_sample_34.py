# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import sample_34


G = nx.karate_club_graph()
first_edge_result = list(G.edges)[0]
assert get_first_edge(G) is not None and first_edge_result == get_first_edge(G)
