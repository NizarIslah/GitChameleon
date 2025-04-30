import importlib.util
import io
import sys
import unittest
from unittest.mock import patch
import os

# Change here: only go up one directory if you truly need to, or stay in the same directory
# if sample_324.py is located in the same folder as this test file.
# For example, if sample_324.py is located in the same directory as this test, use:
dir_path = os.path.dirname(os.path.abspath(__file__))

# Import sample_324.py from the same directory:
spec = importlib.util.spec_from_file_location("sample_324", os.path.join(dir_path, "sample_324.py"))
sample_324 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample_324)

class TestSample324(unittest.TestCase):
    
    def test_infinite_generator(self):
        """Test that the infinite generator yields values from 0 to 999 and then stops."""
        generator = sample_324.infinite()
        values = list(generator)
        
        # Check that the generator yields exactly 1000 values
        self.assertEqual(len(values), 1000)
        
        # Check that the values start at 0 and end at 999
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 999)
        
        # Check that the values are sequential
        for i in range(1000):
            self.assertEqual(values[i], i)
    
    def test_sol_dict_total(self):
        """Test that sol_dict['total'] is set to infinity."""
        self.assertEqual(sample_324.sol_dict['total'], float('inf'))
    
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_progress_bar_creation(self, mock_stdout):
        """Test that the progress bar is created with the correct total."""
        from tqdm import tqdm

        # Run a small portion of the progress bar to test its creation
        with patch.object(sample_324, 'infinite', return_value=range(5)):
            progress_bar = tqdm(sample_324.infinite(), total=sample_324.sol_dict['total'])
            for progress in progress_bar:
                progress_bar.set_description(f"Processing {progress}")
        
        # Check that the output contains expected progress bar elements
        output = mock_stdout.getvalue()
        self.assertIn("Processing", output)
        
        # The progress bar should be created with infinity as its total
        self.assertEqual(progress_bar.total, float('inf'))
    
    def test_progress_bar_description(self):
        """Test that the progress bar description is set correctly."""
        from tqdm import tqdm

        # Create a progress bar with a small range
        test_range = range(5)
        progress_bar = tqdm(test_range, total=len(test_range))
        
        # Set the description for a specific value
        test_value = 3
        progress_bar.set_description(f"Processing {test_value}")
        
        # Check that the description was set correctly
        self.assertEqual(progress_bar.desc, f"Processing {test_value}")
        
        # Clean up
        progress_bar.close()

if __name__ == '__main__':
    unittest.main()