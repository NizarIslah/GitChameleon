import unittest
import sys
import os
import nltk

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_94 import align_words_func

class TestAlignWordsFunc(unittest.TestCase):
    
    def setUp(self):
        # Download necessary NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def test_exact_match(self):
        """Test with identical hypothesis and reference, expecting all matches."""
        hypothesis = ['this', 'is', 'a', 'test']
        reference = ['this', 'is', 'a', 'test']
        
        matches, h_unmatched, r_unmatched = align_words_func(hypothesis, reference)
        
        # All words should match
        self.assertEqual(len(matches), 4)
        # No unmatched words
        self.assertEqual(len(h_unmatched), 0)
        self.assertEqual(len(r_unmatched), 0)
        
        # Each matched index pair should align identical words
        for (h_idx, r_idx) in matches:
            self.assertEqual(hypothesis[h_idx], reference[r_idx])
    
    def test_partial_match(self):
        """Test with partially matching hypothesis and reference."""
        hypothesis = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        reference = ['the', 'cat', 'is', 'sitting', 'on', 'the', 'mat']
        
        matches, h_unmatched, r_unmatched = align_words_func(hypothesis, reference)
        
        # There should be at least some matches
        self.assertTrue(len(matches) > 0)
        
        # Verify the matched words are actually matching in text
        for (h_idx, r_idx) in matches:
            self.assertEqual(hypothesis[h_idx], reference[r_idx])
    
    def test_no_match(self):
        """Test with completely different hypothesis and reference, expecting no matches."""
        hypothesis = ['apple', 'banana', 'cherry']
        reference = ['dog', 'cat', 'mouse']
        
        matches, h_unmatched, r_unmatched = align_words_func(hypothesis, reference)
        
        # Expect no matched pairs
        self.assertEqual(len(matches), 0)
        # All hypothesis words should be unmatched
        self.assertEqual(len(h_unmatched), len(hypothesis))
        # All reference words should be unmatched
        self.assertEqual(len(r_unmatched), len(reference))
    
    def test_empty_inputs(self):
        """Test with both hypothesis and reference empty."""
        hypothesis = []
        reference = []
        
        matches, h_unmatched, r_unmatched = align_words_func(hypothesis, reference)
        
        # No matches or unmatched words
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(h_unmatched), 0)
        self.assertEqual(len(r_unmatched), 0)
    
    def test_one_empty_input(self):
        """Test with one empty input; then reverse the scenario."""
        hypothesis = ['this', 'is', 'a', 'test']
        reference = []
        
        matches, h_unmatched, r_unmatched = align_words_func(hypothesis, reference)
        
        # No matches
        self.assertEqual(len(matches), 0)
        # All hypothesis words should be unmatched
        self.assertEqual(len(h_unmatched), len(hypothesis))
        # No reference words
        self.assertEqual(len(r_unmatched), 0)
        
        # Reverse
        hypothesis = []
        reference = ['this', 'is', 'a', 'test']
        
        matches, h_unmatched, r_unmatched = align_words_func(hypothesis, reference)
        
        # No matches
        self.assertEqual(len(matches), 0)
        # No hypothesis words
        self.assertEqual(len(h_unmatched), 0)
        # All reference words unmatched
        self.assertEqual(len(r_unmatched), len(reference))

if __name__ == '__main__':
    unittest.main()