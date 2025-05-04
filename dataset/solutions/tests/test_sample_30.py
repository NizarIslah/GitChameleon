# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import sample_30

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check networkx version
nx_version = nx.__version__
print(f"Using networkx version: {nx_version}")


class TestBoundingDistance(unittest.TestCase):
    """Test cases for the bounding_distance function in sample_30.py."""

    def test_simple_connected_graph(self):
        """Test with a simple connected graph."""
        # Create a simple path graph with 5 nodes
        # The diameter of this graph is 4 (from node 0 to node 4)
        G = nx.path_graph(5)

        # Get the bounding distance
        distance = sample_30.bounding_distance(G)

        # Check that we get an integer
        self.assertIsInstance(distance, int)
        # The diameter should be 4
        self.assertEqual(distance, 4)

    def test_graph_with_multiple_components(self):
        """Test with a graph that has multiple disconnected components."""
        # Create a graph with two disconnected components
        G = nx.Graph()
        # Component 1: path graph with 3 nodes
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        # Component 2: path graph with 3 nodes
        G.add_edge(3, 4)
        G.add_edge(4, 5)

        try:
            # Get the bounding distance
            distance = sample_30.bounding_distance(G)

            # If we get here, the function didn't raise an error
            # The result should be infinity or a very large number
            # since the graph is disconnected
            self.assertTrue(distance == float("inf") or distance > 1000)
        except (nx.NetworkXError, ValueError) as e:
            # Some versions of networkx might raise an error for disconnected graphs
            # This is acceptable behavior
            self.assertTrue(
                "disconnected" in str(e)
                or "Infinite" in str(e)
                or "connected" in str(e)
            )

    def test_empty_graph(self):
        """Test with an empty graph."""
        G = nx.Graph()

        try:
            # Get the bounding distance
            distance = sample_30.bounding_distance(G)

            # If we get here, the function didn't raise an error
            # The result should be 0 for an empty graph
            self.assertEqual(distance, 0)
        except (nx.NetworkXError, ValueError) as e:
            # Some versions of networkx might raise an error for empty graphs
            # This is acceptable behavior
            self.assertTrue(
                "empty" in str(e) or "at least one" in str(e) or "Empty" in str(e)
            )

    def test_graph_with_single_node(self):
        """Test with a graph that has a single node."""
        G = nx.Graph()
        G.add_node(0)

        # Get the bounding distance
        distance = sample_30.bounding_distance(G)

        # Check that we get an integer
        self.assertIsInstance(distance, int)
        # The diameter should be 0 for a graph with a single node
        self.assertEqual(distance, 0)

    def test_graph_with_no_edges(self):
        """Test with a graph that has nodes but no edges."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i)

        try:
            # Get the bounding distance
            distance = sample_30.bounding_distance(G)

            # If we get here, the function didn't raise an error
            # The result should be infinity or a very large number
            # since the graph is disconnected
            self.assertTrue(distance == float("inf") or distance > 1000)
        except (nx.NetworkXError, ValueError) as e:
            # Some versions of networkx might raise an error for disconnected graphs
            # This is acceptable behavior
            self.assertTrue(
                "disconnected" in str(e)
                or "Infinite" in str(e)
                or "connected" in str(e)
            )

    def test_directed_graph_input(self):
        """Test with a directed graph input."""
        # Create a directed path graph with 5 nodes
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)

        try:
            # Get the bounding distance
            distance = sample_30.bounding_distance(G)

            # If we get here, the function didn't raise an error
            # The result should be an integer
            self.assertIsInstance(distance, int)
            # The diameter should be 4 for this directed path graph
            self.assertEqual(distance, 4)
        except (nx.NetworkXError, TypeError) as e:
            # Some versions of networkx might raise an error for directed graphs
            # This is acceptable behavior
            self.assertTrue(
                "directed" in str(e)
                or "DiGraph" in str(e)
                or "not a graph" in str(e)
                or "not strongly connected" in str(e)
                or "infinite path length" in str(e)
                or "not connected" in str(e)
            )

    def test_non_graph_input(self):
        """Test with a non-graph input (should raise TypeError)."""
        # Try with a list instead of a graph
        G = [0, 1, 2, 3, 4]

        # This should raise a TypeError or AttributeError
        with self.assertRaises((TypeError, AttributeError)):
            sample_30.bounding_distance(G)

    def test_large_graph_with_long_paths(self):
        """Test with a larger graph that has long paths."""
        # Create a cycle graph with 10 nodes
        # The diameter of this graph is 5 (half the cycle)
        G = nx.cycle_graph(10)

        # Get the bounding distance
        distance = sample_30.bounding_distance(G)

        # Check that we get an integer
        self.assertIsInstance(distance, int)
        # The diameter should be 5
        self.assertEqual(distance, 5)

    def test_complete_graph(self):
        """Test with a complete graph where all nodes are connected."""
        # Create a complete graph with 6 nodes
        # The diameter of a complete graph is always 1
        G = nx.complete_graph(6)

        # Get the bounding distance
        distance = sample_30.bounding_distance(G)

        # Check that we get an integer
        self.assertIsInstance(distance, int)
        # The diameter should be 1
        self.assertEqual(distance, 1)


if __name__ == "__main__":
    unittest.main()
