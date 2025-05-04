import unittest
import spacy
from spacy.pipeline.span_ruler import SpanRuler
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_91 import get_labels

nlp = spacy.blank("en")
ruler = SpanRuler(nlp)

patterns = [
    {"label": "PERSON", "pattern": [{"LOWER": "john"}]},
    {"label": "GPE", "pattern": [{"LOWER": "london"}]},
]
ruler.add_patterns(patterns)
labels = get_labels(ruler)
assert isinstance(labels, tuple)
expected = ("GPE", "PERSON")
assert labels == expected
