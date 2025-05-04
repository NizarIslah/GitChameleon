# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
from sample_31 import naive_modularity_communities

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check networkx version
nx_version = nx.__version__
print(f"Using networkx version: {nx_version}")


class TestNaiveModularityCommunities(unittest.TestCase):
    def test_simple_connected_graph(self):
        """Test with a simple connected graph."""
        G = nx.Graph()
        # Community 1
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        # Community 2
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])
        # Bridge between communities
        G.add_edge(2, 3)

        communities = list(naive_modularity_communities(G))
        self.assertIsInstance(communities, list)

    def test_disconnected_graph(self):
        """Test with a graph that has multiple disconnected components."""
        G = nx.Graph()
        # Component 1
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        # Component 2
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])
        communities = list(naive_modularity_communities(G))
        self.assertIsInstance(communities, list)

    def test_directed_graph_input(self):
        """Test with a directed graph input."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
        communities = list(naive_modularity_communities(G))
        self.assertIsInstance(communities, list)

    def test_complete_graph(self):
        """Test with a complete graph where all nodes are connected."""
        G = nx.complete_graph(6)
        communities = list(naive_modularity_communities(G))
        self.assertIsInstance(communities, list)

    def test_known_community_structure(self):
        """Test with a graph that has a known community structure."""
        G = nx.Graph()
        # Clique 1
        for i in range(5):
            for j in range(i + 1, 5):
                G.add_edge(i, j)
        # Clique 2
        for i in range(5, 10):
            for j in range(i + 1, 10):
                G.add_edge(i, j)
        # Bridge between cliques
        G.add_edge(4, 5)

        communities = list(naive_modularity_communities(G))
        self.assertIsInstance(communities, list)


if __name__ == "__main__":
    unittest.main()
