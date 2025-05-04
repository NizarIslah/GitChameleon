import pytest
import spacy
from spacy.pipeline.span_ruler import SpanRuler
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_93 import remove_pattern_by_id

nlp = spacy.blank("en")
ruler = SpanRuler(nlp)

patterns = [
    {"label": "PERSON", "pattern": [{"LOWER": "john"}], "id": "pattern1"},
    {"label": "GPE", "pattern": [{"LOWER": "london"}], "id": "pattern2"},
]
ruler.add_patterns(patterns)

assert len(ruler.patterns) == 2

pattern_id_to_remove = "pattern1"

remove_pattern_by_id(ruler, pattern_id_to_remove)
assert len(ruler.patterns) == 1
remaining_pattern_ids = [pattern["id"] for pattern in ruler.patterns]
assert pattern_id_to_remove not in remaining_pattern_ids
