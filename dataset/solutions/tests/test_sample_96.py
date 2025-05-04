import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_96
from sample_96 import parse_sinica_treebank_sentence
from nltk.tree import Tree


class TestParseSinicaTreebankSentence(unittest.TestCase):
    def test_empty_sentence(self):
        """Test that an empty sentence raises an appropriate exception."""
        sentence = ""
        # Define the global that the function uses internally
        sample_96.tree_string = sentence
        with self.assertRaises(ValueError):
            parse_sinica_treebank_sentence(sentence)

    def test_invalid_sentence_format(self):
        """Test that an invalid sentence format raises an appropriate exception."""
        sentence = "This is not a valid tree format"
        sample_96.tree_string = sentence
        with self.assertRaises(ValueError):
            parse_sinica_treebank_sentence(sentence)

    def test_parse_sinica_treebank_sentence_returns_tree(self):
        """Test that parse_sinica_treebank_sentence returns a Tree object."""
        sentence = "(S (NP (Nba 政府)) (VP (VHC 宣布) (S (NP (Nba 今天)) (VP (VH11 是) (NP (Ncb 國定) (Nab 假日))))))"
        sample_96.tree_string = sentence
        result = parse_sinica_treebank_sentence(sentence)
        self.assertIsInstance(result, Tree)

    def test_parse_sinica_treebank_sentence_leaves(self):
        """Test that the parsed tree has the correct leaves."""
        sentence = "(S (NP (Nba 政府)) (VP (VHC 宣布) (S (NP (Nba 今天)) (VP (VH11 是) (NP (Ncb 國定) (Nab 假日))))))"
        sample_96.tree_string = sentence
        result = parse_sinica_treebank_sentence(sentence)
        expected_leaves = ["政府", "宣布", "今天", "是", "國定", "假日"]
        self.assertEqual(result.leaves(), expected_leaves)

    def test_parse_sinica_treebank_sentence_structure(self):
        """Test that the parsed tree has the correct structure."""
        sentence = "(S (NP (Nba 政府)) (VP (VHC 宣布) (S (NP (Nba 今天)) (VP (VH11 是) (NP (Ncb 國定) (Nab 假日))))))"
        sample_96.tree_string = sentence
        result = parse_sinica_treebank_sentence(sentence)
        # Root label should be 'S'
        self.assertEqual(result.label(), "S")
        # First child should be an NP subtree
        self.assertEqual(result[0].label(), "NP")
        # Second child should be a VP subtree
        self.assertEqual(result[1].label(), "VP")


if __name__ == "__main__":
    unittest.main()
