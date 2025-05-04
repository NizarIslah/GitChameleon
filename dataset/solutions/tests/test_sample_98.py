import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_98 import tokenize_sentence  # adjust module name if needed


class TestTokenizeSentence(unittest.TestCase):
    def test_basic_tokenization(self):
        """Test simple sentence tokenization."""
        sentence = "This is a test."
        tokens = tokenize_sentence(sentence)
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens, ["This", "is", "a", "test", "."])

    def test_contractions(self):
        """Test that contractions are handled correctly."""
        sentence = "I can't do this."
        tokens = tokenize_sentence(sentence)
        self.assertIn("ca", tokens)
        self.assertIn("n't", tokens)

    def test_punctuation(self):
        """Test that punctuation is separated."""
        sentence = "Hello, world!"
        tokens = tokenize_sentence(sentence)
        self.assertIn("Hello", tokens)
        self.assertIn(",", tokens)
        self.assertIn("world", tokens)
        self.assertIn("!", tokens)

    def test_special_characters(self):
        """Test tokenization with special characters."""
        sentence = "The price is $10.99 for a 2-pack at 75% off!"
        tokens = tokenize_sentence(sentence)
        self.assertIsInstance(tokens, list)
        self.assertIn("$", tokens)
        self.assertIn("10.99", tokens)
        # NLTK tokenizes "2-pack" as one token
        self.assertIn("2-pack", tokens)
        self.assertIn("75", tokens)
        self.assertIn("%", tokens)
        self.assertIn("!", tokens)

    def test_empty_string(self):
        """Test that an empty string returns an empty list."""
        tokens = tokenize_sentence("")
        self.assertEqual(tokens, [])


if __name__ == "__main__":
    unittest.main()
