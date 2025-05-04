import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_98 import tokenize_sentence  # adjust module name if needed

sentence = "This is a test sentence."
tokens = tokenize_sentence(sentence)
assert isinstance(tokens, list)
assert tokens == ["This", "is", "a", "test", "sentence", "."]
