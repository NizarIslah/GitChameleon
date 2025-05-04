# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import sample_35

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check networkx version
nx_version = nx.__version__
print(f"Using networkx version: {nx_version}")


class TestShortestPath(unittest.TestCase):
    """Test cases for the shortest_path function in sample_35.py."""

    def test_simple_graph_with_positive_weights(self):
        """Test with a simple graph with positive weights."""
        # Create a simple graph with positive weights
        G = nx.Graph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=2)
        G.add_edge(0, 2, weight=4)

        # Get the shortest paths from node 0
        predecessors, distances = sample_35.shortest_path(G, 0)

        # Check that we get dictionaries
        self.assertIsInstance(predecessors, dict)
        self.assertIsInstance(distances, dict)

        # Check the distances
        self.assertEqual(distances[0], 0)  # Distance to self is 0
        self.assertEqual(distances[1], 1)  # Distance to node 1 is 1
        self.assertEqual(distances[2], 3)  # Distance to node 2 is 3 (via node 1)

        # Check the predecessors
        self.assertEqual(predecessors[0], [])  # No predecessor for source
        self.assertEqual(predecessors[1], [0])  # Predecessor of 1 is 0
        self.assertEqual(predecessors[2], [1])  # Predecessor of 2 is 1

    def test_graph_with_negative_weights(self):
        """Test with a graph that has negative weights."""
        # Create a graph with negative weights
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=-3)
        G.add_edge(0, 2, weight=5)

        # Get the shortest paths from node 0
        predecessors, distances = sample_35.shortest_path(G, 0)

        # Check the distances
        self.assertEqual(distances[0], 0)  # Distance to self is 0
        self.assertEqual(distances[1], 1)  # Distance to node 1 is 1
        self.assertEqual(distances[2], -2)  # Distance to node 2 is -2 (via node 1)

        # Check the predecessors
        self.assertEqual(predecessors[0], [])  # No predecessor for source
        self.assertEqual(predecessors[1], [0])  # Predecessor of 1 is 0
        self.assertEqual(predecessors[2], [1])  # Predecessor of 2 is 1

    def test_directed_graph(self):
        """Test with a directed graph."""
        # Create a directed graph
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=2)
        G.add_edge(2, 0, weight=3)  # This creates a cycle

        # Get the shortest paths from node 0
        predecessors, distances = sample_35.shortest_path(G, 0)

        # Check the distances
        self.assertEqual(distances[0], 0)  # Distance to self is 0
        self.assertEqual(distances[1], 1)  # Distance to node 1 is 1
        self.assertEqual(distances[2], 3)  # Distance to node 2 is 3 (via node 1)

        # Check the predecessors
        self.assertEqual(predecessors[0], [])  # No predecessor for source
        self.assertEqual(predecessors[1], [0])  # Predecessor of 1 is 0
        self.assertEqual(predecessors[2], [1])  # Predecessor of 2 is 1

    def test_graph_with_disconnected_components(self):
        """Test with a graph that has disconnected components."""
        # Create a graph with disconnected components
        G = nx.Graph()
        # First component
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=2)
        # Second component (disconnected)
        G.add_edge(3, 4, weight=3)
        G.add_edge(4, 5, weight=4)

        # Get the shortest paths from node 0
        predecessors, distances = sample_35.shortest_path(G, 0)

        # Check the distances for connected nodes
        self.assertEqual(distances[0], 0)  # Distance to self is 0
        self.assertEqual(distances[1], 1)  # Distance to node 1 is 1
        self.assertEqual(distances[2], 3)  # Distance to node 2 is 3 (via node 1)

        # Check that disconnected nodes are not in the distances dictionary
        # or have infinite distance (depending on NetworkX version)
        for node in [3, 4, 5]:
            if node in distances:
                self.assertEqual(distances[node], float("inf"))

        # Check the predecessors
        self.assertEqual(predecessors[0], [])  # No predecessor for source
        self.assertEqual(predecessors[1], [0])  # Predecessor of 1 is 0
        self.assertEqual(predecessors[2], [1])  # Predecessor of 2 is 1
        # Disconnected nodes might not be in the predecessors dictionary
        # or have empty predecessor lists (depending on NetworkX version)
        for node in [3, 4, 5]:
            if node in predecessors:
                self.assertEqual(predecessors[node], [])

    def test_graph_with_negative_cycle(self):
        """Test with a graph that has a negative cycle (should raise NetworkXUnbounded)."""
        # Create a graph with a negative cycle
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=2)
        G.add_edge(2, 0, weight=-4)  # This creates a negative cycle

        # This should raise NetworkXUnbounded
        with self.assertRaises(nx.NetworkXUnbounded):
            sample_35.shortest_path(G, 0)

    def test_empty_graph(self):
        """Test with an empty graph."""
        # Create an empty graph
        G = nx.Graph()
        # Add node 0 to the empty graph
        G.add_node(0)

        # Get the shortest paths from node 0
        predecessors, distances = sample_35.shortest_path(G, 0)

        # Check that we get dictionaries with only the source node
        self.assertEqual(predecessors, {0: []})
        self.assertEqual(distances, {0: 0})

    def test_non_graph_input(self):
        """Test with a non-graph input (should raise TypeError or AttributeError)."""
        # Try with a list instead of a graph
        G = [0, 1, 2, 3, 4]

        # This should raise a TypeError or AttributeError
        with self.assertRaises((TypeError, AttributeError)):
            sample_35.shortest_path(G, 0)

    def test_invalid_source_node(self):
        """Test with an invalid source node (should raise NodeNotFound)."""
        # Create a simple graph
        G = nx.Graph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=2)

        # Try with a source node that doesn't exist in the graph
        with self.assertRaises(nx.NodeNotFound):
            sample_35.shortest_path(G, 3)


if __name__ == "__main__":
    unittest.main()
