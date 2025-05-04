import unittest
import sys
import os
import nltk
from nltk.corpus import wordnet

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_95 import get_synset_examples

synset = "dog.n.01"
examples = get_synset_examples(synset)
assert isinstance(examples, list)
assert examples == ["the dog barked all night"]
