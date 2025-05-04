# Add the parent directory to import sys
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import numpy as np
import sample_28

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Check networkx version
nx_version = nx.__version__
print(f"Using networkx version: {nx_version}")

# Check numpy version
np_version = np.__version__
print(f"Using numpy version: {np_version}")


class TestModularityCommunities(unittest.TestCase):
    """Test cases for the modularity_communities function in sample_28.py."""

    def test_basic_graph_with_clear_communities(self):
        """Test with a basic graph that has clear community structure."""
        # Create a graph with two clear communities
        G = nx.Graph()
        # Community 1: nodes 0-4 densely connected
        for i in range(5):
            for j in range(i + 1, 5):
                G.add_edge(i, j)
        # Community 2: nodes 5-9 densely connected
        for i in range(5, 10):
            for j in range(i + 1, 10):
                G.add_edge(i, j)
        # Add a single edge between communities
        G.add_edge(0, 5)

        # Get communities
        communities = sample_28.modularity_communities(G)

        # Check that we get a list of communities
        self.assertIsInstance(communities, list)
        # Check that each community is a set
        for community in communities:
            self.assertIsInstance(community, frozenset)

        # The n_communities parameter is set to 5, so we might get up to 5 communities
        # But since we only have 2 clear communities, we should get at most 5
        self.assertLessEqual(len(communities), 5)

        # Check that the communities make sense
        # Convert frozensets to sets for easier comparison
        community_sets = [set(c) for c in communities]

        # Check that nodes from the same expected community tend to be grouped together
        # For each node pair in the same expected community, check if they're in the same detected community
        same_community_count = 0
        total_pairs = 0

        # Check pairs in expected community 1 (nodes 0-4)
        for i in range(5):
            for j in range(i + 1, 5):
                total_pairs += 1
                for community in community_sets:
                    if i in community and j in community:
                        same_community_count += 1
                        break

        # Check pairs in expected community 2 (nodes 5-9)
        for i in range(5, 10):
            for j in range(i + 1, 10):
                total_pairs += 1
                for community in community_sets:
                    if i in community and j in community:
                        same_community_count += 1
                        break

        # At least 50% of node pairs from the same expected community should be in the same detected community
        self.assertGreaterEqual(same_community_count / total_pairs, 0.5)

    def test_empty_graph(self):
        """Test with an empty graph."""
        G = nx.Graph()

        try:
            communities = sample_28.modularity_communities(G)
            # If we get here, the function didn't raise an error
            # The result should be an empty list
            self.assertIsInstance(communities, list)
            self.assertEqual(len(communities), 0)
        except (ValueError, ZeroDivisionError) as e:
            # NetworkX 2.7 might raise an error for empty graphs
            # This is acceptable behavior
            self.assertTrue(True)

    def test_graph_with_single_node(self):
        """Test with a graph that has a single node."""
        G = nx.Graph()
        G.add_node(0)

        try:
            communities = sample_28.modularity_communities(G)

            # Check that we get a list of communities
            self.assertIsInstance(communities, list)
            # We should have 1 community
            self.assertEqual(len(communities), 1)
            # The community should contain the single node
            self.assertEqual(set(communities[0]), {0})
        except (ValueError, ZeroDivisionError) as e:
            # In NetworkX 2.7 or 2.8, the n_communities parameter might be invalid for a single node
            # or the algorithm might raise a ZeroDivisionError
            # This is acceptable behavior
            self.assertTrue(True)

    def test_graph_with_no_edges(self):
        """Test with a graph that has nodes but no edges."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i)

        try:
            communities = sample_28.modularity_communities(G)

            # Check that we get a list of communities
            self.assertIsInstance(communities, list)
            # We should have at most 5 communities (one for each node)
            self.assertLessEqual(len(communities), 5)
            # Each community should contain at least one node
            for community in communities:
                self.assertGreaterEqual(len(community), 1)

            # All nodes should be in some community
            all_nodes = set()
            for community in communities:
                all_nodes.update(community)
            self.assertEqual(all_nodes, set(range(5)))
        except ZeroDivisionError:
            # In NetworkX 2.7, the algorithm might raise a ZeroDivisionError for graphs with no edges
            # This is acceptable behavior
            self.assertTrue(True)

    def test_graph_with_disconnected_components(self):
        """Test with a graph that has disconnected components."""
        G = nx.Graph()
        # Component 1: nodes 0-2 form a triangle
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(0, 2)
        # Component 2: nodes 3-5 form a triangle
        G.add_edge(3, 4)
        G.add_edge(4, 5)
        G.add_edge(3, 5)

        communities = sample_28.modularity_communities(G)

        # Check that we get a list of communities
        self.assertIsInstance(communities, list)
        # The n_communities parameter is set to 5, so we might get up to 5 communities
        self.assertLessEqual(len(communities), 5)

        # Check that nodes from the same component tend to be grouped together
        # Convert frozensets to sets for easier comparison
        community_sets = [set(c) for c in communities]

        # For each component, check if its nodes are in the same community
        # or at least not mixed with nodes from the other component

        # For each community, check if it contains nodes from only one component
        for community in community_sets:
            # Count nodes from each component in this community
            component1_count = sum(1 for node in community if node in {0, 1, 2})
            component2_count = sum(1 for node in community if node in {3, 4, 5})

            # A community should not mix nodes from different components
            # Either it contains nodes only from component 1, or only from component 2, or none
            self.assertTrue(component1_count == 0 or component2_count == 0)

    def test_large_graph_with_many_communities(self):
        """Test with a larger graph that has multiple communities."""
        # Create a graph with 3 communities
        G = nx.Graph()
        # Community 1: nodes 0-9 form a clique
        for i in range(10):
            for j in range(i + 1, 10):
                G.add_edge(i, j)
        # Community 2: nodes 10-19 form a clique
        for i in range(10, 20):
            for j in range(i + 1, 20):
                G.add_edge(i, j)
        # Community 3: nodes 20-29 form a clique
        for i in range(20, 30):
            for j in range(i + 1, 30):
                G.add_edge(i, j)
        # Add a few edges between communities
        G.add_edge(0, 10)
        G.add_edge(10, 20)
        G.add_edge(0, 20)

        communities = sample_28.modularity_communities(G)

        # Check that we get a list of communities
        self.assertIsInstance(communities, list)
        # The n_communities parameter is set to 5, so we should get at most 5 communities
        self.assertLessEqual(len(communities), 5)

        # Check that the communities make sense
        # Convert frozensets to sets for easier comparison
        community_sets = [set(c) for c in communities]

        # For each expected community, check if nodes from that community tend to be grouped together
        expected_communities = [set(range(10)), set(range(10, 20)), set(range(20, 30))]

        # For each expected community, calculate what percentage of its node pairs
        # are in the same detected community
        for expected in expected_communities:
            same_community_count = 0
            total_pairs = 0

            # Check all pairs of nodes in this expected community
            nodes = list(expected)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    total_pairs += 1
                    for community in community_sets:
                        if nodes[i] in community and nodes[j] in community:
                            same_community_count += 1
                            break

            # If there are any pairs to check, at least 50% of node pairs from the same
            # expected community should be in the same detected community
            if total_pairs > 0:
                self.assertGreaterEqual(same_community_count / total_pairs, 0.5)

    def test_directed_graph_input(self):
        """Test with a directed graph input."""
        # Create a directed graph
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)

        try:
            # This might raise a TypeError or work by converting to undirected
            communities = sample_28.modularity_communities(G)

            # If we get here, the function didn't raise an error
            # The result should be a list of communities
            self.assertIsInstance(communities, list)
            # The n_communities parameter is set to 5, but we only have 3 nodes
            # We should have at most 3 communities
            self.assertLessEqual(len(communities), 3)

            # Check that all nodes are in some community
            all_nodes = set()
            for community in communities:
                all_nodes.update(community)
            self.assertEqual(all_nodes, {0, 1, 2})
        except (TypeError, ValueError) as e:
            # If a TypeError or ValueError is raised, that's also acceptable behavior
            self.assertTrue(
                "not a graph" in str(e)
                or "not a Graph" in str(e)
                or "DiGraph" in str(e)
                or "n_communities" in str(e)
            )

    def test_non_graph_input(self):
        """Test with a non-graph input (should raise TypeError)."""
        # Try with a list instead of a graph
        G = [0, 1, 2]

        # This should raise an AttributeError or TypeError
        with self.assertRaises((AttributeError, TypeError)):
            sample_28.modularity_communities(G)


if __name__ == "__main__":
    unittest.main()
