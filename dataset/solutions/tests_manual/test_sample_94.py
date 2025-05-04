import unittest
import sys
import os
import nltk

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_94 import align_words_func

hypothesis = ["the", "cat", "sits", "on", "the", "mat"]
reference = ["the", "cat", "is", "sitting", "on", "the", "mat"]
expected_matches = [(0, 0), (1, 1), (2, 3), (3, 4), (4, 5), (5, 6)]
matches, unmatched_hypo, unmatched_ref = align_words_func(hypothesis, reference)
assert matches == expected_matches
assert unmatched_hypo == []
assert unmatched_ref == [(2, "is")]
