import unittest
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_98 import tokenize_sentence

class TestTokenizeSentence(unittest.TestCase):
    
    def test_basic_sentence(self):
        """Test tokenization of a basic sentence."""
        sentence = "This is a simple test sentence."
        tokens = tokenize_sentence(sentence)
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens, ['This', 'is', 'a', 'simple', 'test', 'sentence', '.'])
    
    def test_empty_string(self):
        """Test tokenization of an empty string."""
        sentence = ""
        tokens = tokenize_sentence(sentence)
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens, [])
    
    def test_punctuation(self):
        """Test tokenization with various punctuation marks."""
        sentence = "Hello, world! How are you? I'm fine; thanks."
        tokens = tokenize_sentence(sentence)
        self.assertIsInstance(tokens, list)
        expected = ['Hello', ',', 'world', '!', 'How', 'are', 'you', '?', 'I', "'m", 'fine', ';', 'thanks', '.']
        self.assertEqual(tokens, expected)
    
    def test_contractions(self):
        """Test tokenization with contractions."""
        sentence = "I can't believe it's not butter."
        tokens = tokenize_sentence(sentence)
        self.assertIsInstance(tokens, list)
        expected = ['I', 'ca', "n't", 'believe', 'it', "'s", 'not', 'butter', '.']
        self.assertEqual(tokens, expected)
    
    def test_special_characters(self):
        """Test tokenization with special characters."""
        sentence = "The price is $10.99 for a 2-pack at 75% off!"
        tokens = tokenize_sentence(sentence)
        self.assertIsInstance(tokens, list)
        # The exact tokenization may vary depending on NLTK's implementation
        # This is an approximation
        self.assertIn('$', tokens)
        self.assertIn('10.99', tokens)
        self.assertIn('2', tokens)
        self.assertIn('pack', tokens)
        self.assertIn('%', tokens)
    
    def test_multiple_spaces(self):
        """Test tokenization with multiple spaces."""
        sentence = "This   has    multiple    spaces."
        tokens = tokenize_sentence(sentence)
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens, ['This', 'has', 'multiple', 'spaces', '.'])

if __name__ == '__main__':
    unittest.main()