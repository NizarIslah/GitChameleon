import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_96
from sample_96 import parse_sinica_treebank_sentence
from nltk.tree import Tree

sinica_sentence = sinica_treebank.parsed_sents()[0]
tree_string = sinica_sentence.pformat()

parsed_tree = parse_sinica_treebank_sentence(tree_string)
assert isinstance(parsed_tree, Tree)
assert parsed_tree.label() == "NP"
