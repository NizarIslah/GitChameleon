# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import sample_34

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check networkx version
nx_version = nx.__version__
print(f"Using networkx version: {nx_version}")


class TestGetFirstEdge(unittest.TestCase):
    """Test cases for the get_first_edge function in sample_34.py."""

    def test_simple_graph(self):
        """Test with a simple graph with a single edge."""
        # Create a simple graph with a single edge
        G = nx.Graph()
        G.add_edge(0, 1)

        # Get the first edge
        edge = sample_34.get_first_edge(G)

        # Check that we get a tuple
        self.assertIsInstance(edge, tuple)
        # Check that the tuple contains the correct nodes
        self.assertEqual(set(edge), {0, 1})
        # Check that the tuple has the correct length
        self.assertEqual(len(edge), 2)

    def test_graph_with_multiple_edges(self):
        """Test with a graph that has multiple edges."""
        G = nx.Graph()
        # Add multiple edges
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        # Get the first edge
        edge = sample_34.get_first_edge(G)

        # Check that we get a tuple
        self.assertIsInstance(edge, tuple)
        # Check that the tuple contains valid nodes
        self.assertTrue(edge[0] in G.nodes and edge[1] in G.nodes)
        # Check that the tuple has the correct length
        self.assertEqual(len(edge), 2)
        # Check that the edge exists in the graph
        self.assertTrue(G.has_edge(edge[0], edge[1]))

    def test_directed_graph(self):
        """Test with a directed graph."""
        G = nx.DiGraph()
        # Add edges
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        # Get the first edge
        edge = sample_34.get_first_edge(G)

        # Check that we get a tuple
        self.assertIsInstance(edge, tuple)
        # Check that the tuple contains valid nodes
        self.assertTrue(edge[0] in G.nodes and edge[1] in G.nodes)
        # Check that the tuple has the correct length
        self.assertEqual(len(edge), 2)
        # Check that the edge exists in the graph
        self.assertTrue(G.has_edge(edge[0], edge[1]))
        # For directed graphs, the order of nodes in the edge matters
        # The first node should be the source and the second node should be the target
        self.assertTrue(G.has_edge(edge[0], edge[1]))

    def test_multigraph(self):
        """Test with a multigraph (graph that allows multiple edges between nodes)."""
        G = nx.MultiGraph()
        # Add edges
        G.add_edge(0, 1, key=0)
        G.add_edge(0, 1, key=1)  # Duplicate edge with different key
        G.add_edge(1, 2)

        # Get the first edge
        edge = sample_34.get_first_edge(G)

        # Check that we get a tuple
        self.assertIsInstance(edge, tuple)
        # For multigraphs, the edge might be a 3-tuple (u, v, key)
        # or a 2-tuple (u, v) depending on the NetworkX version
        self.assertTrue(len(edge) in [2, 3])
        # Check that the first two elements are valid nodes
        self.assertTrue(edge[0] in G.nodes and edge[1] in G.nodes)
        # Check that the edge exists in the graph
        self.assertTrue(G.has_edge(edge[0], edge[1]))

    def test_graph_with_string_nodes(self):
        """Test with a graph that has string nodes."""
        G = nx.Graph()
        # Add edges with string nodes
        G.add_edge("A", "B")
        G.add_edge("B", "C")

        # Get the first edge
        edge = sample_34.get_first_edge(G)

        # Check that we get a tuple
        self.assertIsInstance(edge, tuple)
        # Check that the tuple contains valid nodes
        self.assertTrue(edge[0] in G.nodes and edge[1] in G.nodes)
        # Check that the tuple has the correct length
        self.assertEqual(len(edge), 2)
        # Check that the edge exists in the graph
        self.assertTrue(G.has_edge(edge[0], edge[1]))

    def test_graph_with_mixed_node_types(self):
        """Test with a graph that has mixed node types."""
        G = nx.Graph()
        # Add edges with mixed node types
        G.add_edge(0, "A")
        G.add_edge("A", 2.5)
        G.add_edge(2.5, True)

        # Get the first edge
        edge = sample_34.get_first_edge(G)

        # Check that we get a tuple
        self.assertIsInstance(edge, tuple)
        # Check that the tuple contains valid nodes
        self.assertTrue(edge[0] in G.nodes and edge[1] in G.nodes)
        # Check that the tuple has the correct length
        self.assertEqual(len(edge), 2)
        # Check that the edge exists in the graph
        self.assertTrue(G.has_edge(edge[0], edge[1]))

    def test_empty_graph(self):
        """Test with an empty graph (should raise IndexError)."""
        G = nx.Graph()

        # This should raise an IndexError
        with self.assertRaises(IndexError):
            sample_34.get_first_edge(G)

    def test_graph_with_no_edges(self):
        """Test with a graph that has nodes but no edges (should raise IndexError)."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i)

        # This should raise an IndexError
        with self.assertRaises(IndexError):
            sample_34.get_first_edge(G)

    def test_non_graph_input(self):
        """Test with a non-graph input (should raise TypeError or AttributeError)."""
        # Try with a list instead of a graph
        G = [0, 1, 2, 3, 4]

        # This should raise a TypeError or AttributeError
        with self.assertRaises((TypeError, AttributeError)):
            sample_34.get_first_edge(G)


if __name__ == "__main__":
    unittest.main()
