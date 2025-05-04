# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import sample_29


G = nx.path_graph(5)
result = nx.diameter(G, usebounds=True)
assert bounding_distance(G) is not None and result == bounding_distance(G)
