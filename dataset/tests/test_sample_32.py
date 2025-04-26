import unittest
import sys
import os
import warnings

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
import sample_32

# Filter deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Check networkx version
nx_version = nx.__version__
print(f"Using networkx version: {nx_version}")


class TestNaiveModularityCommunities(unittest.TestCase):
    """Test cases for the naive_modularity_communities function in sample_32.py."""

    def test_simple_connected_graph(self):
        """Test with a simple connected graph."""
        # Create a simple graph with two communities
        G = nx.Graph()
        # Community 1
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        # Community 2
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])
        # Bridge between communities
        G.add_edge(2, 3)
        
        # Get the communities
        communities = sample_32.naive_modularity_communities(G)
        
        # Check that we get a list
        self.assertIsInstance(communities, list)
        # Check that each community is a frozenset
        for community in communities:
            self.assertIsInstance(community, frozenset)
        # Check that we have at least one community
        self.assertGreaterEqual(len(communities), 1)
        # Check that all nodes are in some community
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)
        self.assertEqual(all_nodes, set(G.nodes()))

    def test_empty_graph(self):
        """Test with an empty graph."""
        G = nx.Graph()
        
        try:
            # Get the communities
            communities = sample_32.naive_modularity_communities(G)
            
            # If we get here, the function didn't raise an error
            # Check that we get a list
            self.assertIsInstance(communities, list)
            # The list should be empty for an empty graph
            self.assertEqual(len(communities), 0)
        except (nx.NetworkXError, ValueError, ZeroDivisionError) as e:
            # Some versions of networkx might raise an error for empty graphs
            # This is acceptable behavior
            self.assertTrue("empty" in str(e) or 
                           "at least one" in str(e) or 
                           "Empty" in str(e) or
                           "division by zero" in str(e))

    def test_graph_with_single_node(self):
        """Test with a graph that has a single node."""
        G = nx.Graph()
        G.add_node(0)
        
        try:
            # Get the communities
            communities = sample_32.naive_modularity_communities(G)
            
            # Check that we get a list
            self.assertIsInstance(communities, list)
            # Check that we have exactly one community
            self.assertEqual(len(communities), 1)
            # Check that the community contains the single node
            self.assertEqual(communities[0], frozenset({0}))
        except (nx.NetworkXError, ValueError, ZeroDivisionError) as e:
            # Some versions of networkx might raise an error for graphs with a single node
            # This is acceptable behavior
            self.assertTrue("empty" in str(e) or 
                           "at least one" in str(e) or 
                           "Empty" in str(e) or
                           "division by zero" in str(e))

    def test_graph_with_no_edges(self):
        """Test with a graph that has nodes but no edges."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i)
        
        try:
            # Get the communities
            communities = sample_32.naive_modularity_communities(G)
            
            # Check that we get a list
            self.assertIsInstance(communities, list)
            # Check that we have the right number of communities
            # Each node should be in its own community
            self.assertEqual(len(communities), 5)
            # Check that each community has exactly one node
            for i, community in enumerate(communities):
                self.assertEqual(len(community), 1)
                self.assertTrue(i in community)
        except (nx.NetworkXError, ValueError, ZeroDivisionError) as e:
            # Some versions of networkx might raise an error for graphs with no edges
            # This is acceptable behavior
            self.assertTrue("empty" in str(e) or 
                           "at least one" in str(e) or 
                           "Empty" in str(e) or
                           "division by zero" in str(e))

    def test_directed_graph_input(self):
        """Test with a directed graph input."""
        # Create a directed graph
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
        
        try:
            # Get the communities
            communities = sample_32.naive_modularity_communities(G)
            
            # If we get here, the function didn't raise an error
            # Check that we get a list
            self.assertIsInstance(communities, list)
            # Check that each community is a set
            for community in communities:
                self.assertIsInstance(community, frozenset)
            # Check that all nodes are in some community
            all_nodes = set()
            for community in communities:
                all_nodes.update(community)
            self.assertEqual(all_nodes, set(G.nodes()))
        except (nx.NetworkXError, TypeError, nx.NetworkXNotImplemented) as e:
            # Some versions of networkx might raise an error for directed graphs
            # This is acceptable behavior
            self.assertTrue("directed" in str(e) or 
                           "DiGraph" in str(e) or 
                           "not a graph" in str(e) or
                           "undirected" in str(e) or
                           "not implemented for directed" in str(e))

    def test_non_graph_input(self):
        """Test with a non-graph input (should raise TypeError)."""
        # Try with a list instead of a graph
        G = [0, 1, 2, 3, 4]
        
        # This should raise a TypeError or AttributeError
        with self.assertRaises((TypeError, AttributeError)):
            sample_32.naive_modularity_communities(G)

    def test_complete_graph(self):
        """Test with a complete graph where all nodes are connected."""
        # Create a complete graph with 6 nodes
        G = nx.complete_graph(6)
        
        # Get the communities
        communities = sample_32.naive_modularity_communities(G)
        
        # Check that we get a list
        self.assertIsInstance(communities, list)
        # Check that we have at least one community
        self.assertGreaterEqual(len(communities), 1)
        # Check that all nodes are in some community
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)
        self.assertEqual(all_nodes, set(G.nodes()))

    def test_disconnected_graph(self):
        """Test with a graph that has multiple disconnected components."""
        # Create a graph with two disconnected components
        G = nx.Graph()
        # Component 1: complete graph with 3 nodes
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        # Component 2: complete graph with 3 nodes
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])
        
        # Get the communities
        communities = sample_32.naive_modularity_communities(G)
        
        # Check that we get a list
        self.assertIsInstance(communities, list)
        # Check that we have at least two communities (one for each component)
        self.assertGreaterEqual(len(communities), 2)
        # Check that all nodes are in some community
        all_nodes = set()
        for community in communities:
            all_nodes.update(community)
        self.assertEqual(all_nodes, set(G.nodes()))

    def test_known_community_structure(self):
        """Test with a graph that has a known community structure."""
        # Create a graph with a known community structure
        # Two cliques connected by a single edge
        G = nx.Graph()
        # Clique 1
        for i in range(5):
            for j in range(i+1, 5):
                G.add_edge(i, j)
        # Clique 2
        for i in range(5, 10):
            for j in range(i+1, 10):
                G.add_edge(i, j)
        # Bridge between cliques
        G.add_edge(4, 5)
        
        # Get the communities
        communities = sample_32.naive_modularity_communities(G)
        
        # Check that we get a list
        self.assertIsInstance(communities, list)
        # We should have at least 2 communities
        self.assertGreaterEqual(len(communities), 2)
        
        # Find the communities containing nodes 0 and 9
        community_with_0 = None
        community_with_9 = None
        for community in communities:
            if 0 in community:
                community_with_0 = community
            if 9 in community:
                community_with_9 = community
        
        # Check that nodes 0-4 are in the same community
        self.assertIsNotNone(community_with_0)
        for i in range(1, 5):
            self.assertIn(i, community_with_0)
        
        # Check that nodes 5-9 are in the same community
        self.assertIsNotNone(community_with_9)
        for i in range(5, 9):
            self.assertIn(i, community_with_9)
        
        # Check that the two communities are different
        self.assertNotEqual(community_with_0, community_with_9)


if __name__ == '__main__':
    unittest.main()