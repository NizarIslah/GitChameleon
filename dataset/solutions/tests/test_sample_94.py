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
        """Test with identical hypothesis and reference."""
        hypothesis = ['this', 'is', 'a', 'test']
        reference = ['this', 'is', 'a', 'test']
        
        result = align_words_func(hypothesis, reference)
        
        # Unpack the result tuple
        matches, h_unmatched, r_unmatched = result
        
        # All words should match
        self.assertEqual(len(matches), 4)
        # No unmatched words
        self.assertEqual(len(h_unmatched), 0)
        self.assertEqual(len(r_unmatched), 0)
        
        # Check that each match is correct
        for i, ((h_idx, r_idx), _) in enumerate(matches):
            self.assertEqual(hypothesis[h_idx], reference[r_idx])
    
    def test_partial_match(self):
        """Test with partially matching hypothesis and reference."""
        hypothesis = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        reference = ['the', 'cat', 'is', 'sitting', 'on', 'the', 'mat']
        
        result = align_words_func(hypothesis, reference)
        
        # Unpack the result tuple
        matches, h_unmatched, r_unmatched = result
        
        # Check that we have some matches
        self.assertTrue(len(matches) > 0)
        
        # Verify that matched words are actually matching
        for (h_idx, r_idx), _ in matches:
            self.assertEqual(hypothesis[h_idx], reference[r_idx])
    
    def test_no_match(self):
        """Test with completely different hypothesis and reference."""
        hypothesis = ['apple', 'banana', 'cherry']
        reference = ['dog', 'cat', 'mouse']
        
        result = align_words_func(hypothesis, reference)
        
        # Unpack the result tuple
        matches, h_unmatched, r_unmatched = result
        
        # There should be no exact matches
        self.assertEqual(len([m for m in matches if m[1] == 'exact']), 0)
        
        # All words should be unmatched or have stem/synonym matches
        self.assertEqual(len(h_unmatched) + len([m for m in matches if m[1] != 'exact']), 
                         len(hypothesis))
    
    def test_empty_inputs(self):
        """Test with empty hypothesis and reference."""
        hypothesis = []
        reference = []
        
        result = align_words_func(hypothesis, reference)
        
        # Unpack the result tuple
        matches, h_unmatched, r_unmatched = result
        
        # No matches or unmatched words
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(h_unmatched), 0)
        self.assertEqual(len(r_unmatched), 0)
    
    def test_one_empty_input(self):
        """Test with one empty input."""
        hypothesis = ['this', 'is', 'a', 'test']
        reference = []
        
        result = align_words_func(hypothesis, reference)
        
        # Unpack the result tuple
        matches, h_unmatched, r_unmatched = result
        
        # No matches
        self.assertEqual(len(matches), 0)
        # All hypothesis words should be unmatched
        self.assertEqual(len(h_unmatched), len(hypothesis))
        # No reference words to be unmatched
        self.assertEqual(len(r_unmatched), 0)
        
        # Test the reverse case
        hypothesis = []
        reference = ['this', 'is', 'a', 'test']
        
        result = align_words_func(hypothesis, reference)
        
        # Unpack the result tuple
        matches, h_unmatched, r_unmatched = result
        
        # No matches
        self.assertEqual(len(matches), 0)
        # No hypothesis words to be unmatched
        self.assertEqual(len(h_unmatched), 0)
        # All reference words should be unmatched
        self.assertEqual(len(r_unmatched), len(reference))

if __name__ == '__main__':
    unittest.main()