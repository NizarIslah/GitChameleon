import unittest
import spacy
from spacy.training import Example
import numpy as np
from dataset.solutions.sample_92 import create_whitespace_variant


class TestCreateWhitespaceVariant(unittest.TestCase):
    def setUp(self):
        # Load a small spaCy model for testing
        self.nlp = spacy.blank("en")
        
        # Create a simple training example
        text = "This is a test sentence."
        doc = self.nlp(text)
        
        # Create gold annotations (in this case, just use the same doc)
        gold_doc = self.nlp.make_doc(text)
        
        # Create an Example object
        self.example = Example(doc, gold_doc)
    
    def test_create_whitespace_variant_space(self):
        """Test adding a space whitespace variant."""
        whitespace = " "
        position = 4  # Position to insert whitespace
        
        # Create a whitespace variant
        augmented_example = create_whitespace_variant(
            self.nlp, self.example, whitespace, position
        )
        
        # Verify the augmented example is an Example object
        self.assertIsInstance(augmented_example, Example)
        
        # Verify the text has been modified with whitespace
        original_text = self.example.text
        augmented_text = augmented_example.text
        
        # The augmented text should be longer by the length of the whitespace
        self.assertEqual(len(augmented_text), len(original_text) + len(whitespace))
        
        # Check that the whitespace was inserted at the correct position
        expected_text = original_text[:position] + whitespace + original_text[position:]
        self.assertEqual(augmented_text, expected_text)
    
    def test_create_whitespace_variant_newline(self):
        """Test adding a newline whitespace variant."""
        whitespace = "\n"
        position = 7  # Position to insert whitespace
        
        # Create a whitespace variant
        augmented_example = create_whitespace_variant(
            self.nlp, self.example, whitespace, position
        )
        
        # Verify the text has been modified with a newline
        original_text = self.example.text
        augmented_text = augmented_example.text
        
        # Check that the newline was inserted at the correct position
        expected_text = original_text[:position] + whitespace + original_text[position:]
        self.assertEqual(augmented_text, expected_text)
    
    def test_create_whitespace_variant_tab(self):
        """Test adding a tab whitespace variant."""
        whitespace = "\t"
        position = 10  # Position to insert whitespace
        
        # Create a whitespace variant
        augmented_example = create_whitespace_variant(
            self.nlp, self.example, whitespace, position
        )
        
        # Verify the text has been modified with a tab
        original_text = self.example.text
        augmented_text = augmented_example.text
        
        # Check that the tab was inserted at the correct position
        expected_text = original_text[:position] + whitespace + original_text[position:]
        self.assertEqual(augmented_text, expected_text)
    
    def test_create_whitespace_variant_beginning(self):
        """Test adding whitespace at the beginning of the text."""
        whitespace = "  "  # Two spaces
        position = 0  # Beginning of the text
        
        # Create a whitespace variant
        augmented_example = create_whitespace_variant(
            self.nlp, self.example, whitespace, position
        )
        
        # Verify the text has been modified with whitespace at the beginning
        original_text = self.example.text
        augmented_text = augmented_example.text
        
        # Check that the whitespace was inserted at the beginning
        expected_text = whitespace + original_text
        self.assertEqual(augmented_text, expected_text)
    
    def test_create_whitespace_variant_end(self):
        """Test adding whitespace at the end of the text."""
        whitespace = "  "  # Two spaces
        position = len(self.example.text)  # End of the text
        
        # Create a whitespace variant
        augmented_example = create_whitespace_variant(
            self.nlp, self.example, whitespace, position
        )
        
        # Verify the text has been modified with whitespace at the end
        original_text = self.example.text
        augmented_text = augmented_example.text
        
        # Check that the whitespace was inserted at the end
        expected_text = original_text + whitespace
        self.assertEqual(augmented_text, expected_text)


if __name__ == "__main__":
    unittest.main()