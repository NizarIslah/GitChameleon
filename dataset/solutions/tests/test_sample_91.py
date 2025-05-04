import unittest
import spacy
from spacy.pipeline.span_ruler import SpanRuler
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_91 import get_labels


class TestGetLabels(unittest.TestCase):
    def setUp(self):
        # Create a small spaCy model for testing
        self.nlp = spacy.blank("en")

    def test_get_labels_empty(self):
        # Test with a span ruler that has no patterns
        ruler = SpanRuler(self.nlp, name="span_ruler")
        labels = get_labels(ruler)
        self.assertIsInstance(labels, tuple)
        self.assertEqual(len(labels), 0)

    def test_get_labels_with_patterns(self):
        # Test with a span ruler that has patterns with labels
        ruler = SpanRuler(self.nlp, name="span_ruler")
        patterns = [
            {"pattern": [{"LOWER": "apple"}], "label": "FRUIT"},
            {"pattern": [{"LOWER": "carrot"}], "label": "VEGETABLE"},
        ]
        ruler.add_patterns(patterns)

        labels = get_labels(ruler)
        self.assertIsInstance(labels, tuple)
        self.assertEqual(len(labels), 2)
        self.assertIn("FRUIT", labels)
        self.assertIn("VEGETABLE", labels)

    def test_get_labels_with_duplicate_labels(self):
        # Test with a span ruler that has patterns with duplicate labels
        ruler = SpanRuler(self.nlp, name="span_ruler")
        patterns = [
            {"pattern": [{"LOWER": "apple"}], "label": "FOOD"},
            {"pattern": [{"LOWER": "banana"}], "label": "FOOD"},
            {"pattern": [{"LOWER": "carrot"}], "label": "FOOD"},
        ]
        ruler.add_patterns(patterns)

        labels = get_labels(ruler)
        self.assertIsInstance(labels, tuple)
        self.assertEqual(len(labels), 1)
        self.assertIn("FOOD", labels)


if __name__ == "__main__":
    unittest.main()
