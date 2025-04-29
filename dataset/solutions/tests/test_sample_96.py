import unittest
import sys
import os
import nltk
from nltk.tree import Tree

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_96 import parse_sinica_treebank_sentence

class TestParseSinicaTreebankSentence(unittest.TestCase):
    
    def setUp(self):
        # Ensure the required NLTK data is downloaded
        nltk.download('sinica_treebank', quiet=True)
    
    def test_parse_sinica_treebank_sentence_returns_tree(self):
        """Test that parse_sinica_treebank_sentence returns a Tree object."""
        sample_sentence = "(S (NP (Nba 政府)) (VP (VHC 宣布) (S (NP (Nba 今天)) (VP (VH11 是) (NP (Ncb 國定) (Nab 假日))))))"
        result = parse_sinica_treebank_sentence(sample_sentence)
        self.assertIsInstance(result, Tree)
    
    def test_parse_sinica_treebank_sentence_structure(self):
        """Test that the parsed tree has the correct structure."""
        sample_sentence = "(S (NP (Nba 政府)) (VP (VHC 宣布) (S (NP (Nba 今天)) (VP (VH11 是) (NP (Ncb 國定) (Nab 假日))))))"
        result = parse_sinica_treebank_sentence(sample_sentence)
        
        # Check the root label
        self.assertEqual(result.label(), "S")
        
        # Check that the tree has the expected number of children
        self.assertEqual(len(result), 2)
        
        # Check that the first child is an NP
        self.assertEqual(result[0].label(), "NP")
    
    def test_parse_sinica_treebank_sentence_leaves(self):
        """Test that the parsed tree has the correct leaves."""
        sample_sentence = "(S (NP (Nba 政府)) (VP (VHC 宣布) (S (NP (Nba 今天)) (VP (VH11 是) (NP (Ncb 國定) (Nab 假日))))))"
        result = parse_sinica_treebank_sentence(sample_sentence)
        
        # Check the leaves of the tree
        expected_leaves = ["政府", "宣布", "今天", "是", "國定", "假日"]
        self.assertEqual(result.leaves(), expected_leaves)
    
    def test_invalid_sentence_format(self):
        """Test that an invalid sentence format raises an appropriate exception."""
        invalid_sentence = "This is not a valid tree format"
        with self.assertRaises(ValueError):
            parse_sinica_treebank_sentence(invalid_sentence)
    
    def test_empty_sentence(self):
        """Test that an empty sentence raises an appropriate exception."""
        with self.assertRaises(ValueError):
            parse_sinica_treebank_sentence("")

if __name__ == '__main__':
    unittest.main()