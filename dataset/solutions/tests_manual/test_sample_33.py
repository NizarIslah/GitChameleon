# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import sample_33


G = nx.karate_club_graph()
nodes_result = len(list(G.nodes))
assert (
    get_nodes(G) is not None
    and len(get_nodes(G)) > 0
    and len(get_nodes(G)) == nodes_result
)
