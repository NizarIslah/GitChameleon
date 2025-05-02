# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
from sample_29 import bounding_distance
G = nx.path_graph(5)
result = nx.diameter(G, usebounds=True)
assert bounding_distance(G) is not None and result == bounding_distance(G)