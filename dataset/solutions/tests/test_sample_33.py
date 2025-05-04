# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import sample_33

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check networkx version
nx_version = nx.__version__
print(f"Using networkx version: {nx_version}")


class TestGetNodes(unittest.TestCase):
    """Test cases for the get_nodes function in sample_33.py."""

    def test_simple_graph(self):
        """Test with a simple graph."""
        # Create a simple graph with 5 nodes
        G = nx.Graph()
        for i in range(5):
            G.add_node(i)

        # Get the nodes
        nodes = sample_33.get_nodes(G)

        # Check that we get a list
        self.assertIsInstance(nodes, list)
        # Check that the list contains all the nodes
        self.assertEqual(set(nodes), {0, 1, 2, 3, 4})
        # Check that the list has the correct length
        self.assertEqual(len(nodes), 5)

    def test_empty_graph(self):
        """Test with an empty graph."""
        G = nx.Graph()

        # Get the nodes
        nodes = sample_33.get_nodes(G)

        # Check that we get a list
        self.assertIsInstance(nodes, list)
        # Check that the list is empty
        self.assertEqual(len(nodes), 0)

    def test_graph_with_single_node(self):
        """Test with a graph that has a single node."""
        G = nx.Graph()
        G.add_node(0)

        # Get the nodes
        nodes = sample_33.get_nodes(G)

        # Check that we get a list
        self.assertIsInstance(nodes, list)
        # Check that the list contains the single node
        self.assertEqual(nodes, [0])
        # Check that the list has the correct length
        self.assertEqual(len(nodes), 1)

    def test_graph_with_edges(self):
        """Test with a graph that has nodes and edges."""
        G = nx.Graph()
        # Add nodes and edges
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        # Get the nodes
        nodes = sample_33.get_nodes(G)

        # Check that we get a list
        self.assertIsInstance(nodes, list)
        # Check that the list contains all the nodes
        self.assertEqual(set(nodes), {0, 1, 2})
        # Check that the list has the correct length
        self.assertEqual(len(nodes), 3)

    def test_directed_graph(self):
        """Test with a directed graph."""
        G = nx.DiGraph()
        # Add nodes and edges
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        # Get the nodes
        nodes = sample_33.get_nodes(G)

        # Check that we get a list
        self.assertIsInstance(nodes, list)
        # Check that the list contains all the nodes
        self.assertEqual(set(nodes), {0, 1, 2})
        # Check that the list has the correct length
        self.assertEqual(len(nodes), 3)

    def test_multigraph(self):
        """Test with a multigraph (graph that allows multiple edges between nodes)."""
        G = nx.MultiGraph()
        # Add nodes and edges
        G.add_edge(0, 1)
        G.add_edge(0, 1)  # Duplicate edge
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        # Get the nodes
        nodes = sample_33.get_nodes(G)

        # Check that we get a list
        self.assertIsInstance(nodes, list)
        # Check that the list contains all the nodes
        self.assertEqual(set(nodes), {0, 1, 2})
        # Check that the list has the correct length
        self.assertEqual(len(nodes), 3)

    def test_graph_with_string_nodes(self):
        """Test with a graph that has string nodes."""
        G = nx.Graph()
        # Add nodes
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        # Get the nodes
        nodes = sample_33.get_nodes(G)

        # Check that we get a list
        self.assertIsInstance(nodes, list)
        # Check that the list contains all the nodes
        self.assertEqual(set(nodes), {"A", "B", "C"})
        # Check that the list has the correct length
        self.assertEqual(len(nodes), 3)

    def test_graph_with_mixed_node_types(self):
        """Test with a graph that has mixed node types."""
        G = nx.Graph()
        # Add nodes of different types
        G.add_node(0)
        G.add_node("A")
        G.add_node(2.5)
        G.add_node(True)

        # Get the nodes
        nodes = sample_33.get_nodes(G)

        # Check that we get a list
        self.assertIsInstance(nodes, list)
        # Check that the list contains all the nodes
        self.assertEqual(set(nodes), {0, "A", 2.5, True})
        # Check that the list has the correct length
        self.assertEqual(len(nodes), 4)

    def test_non_graph_input(self):
        """Test with a non-graph input (should raise TypeError or AttributeError)."""
        # Try with a list instead of a graph
        G = [0, 1, 2, 3, 4]

        # This should raise a TypeError or AttributeError
        with self.assertRaises((TypeError, AttributeError)):
            sample_33.get_nodes(G)


if __name__ == "__main__":
    unittest.main()
