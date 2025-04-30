import importlib.util
import io
import sys
import unittest
import os

# Import the module from the solutions directory
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    def test_progress_bar_creation(self):
        """Test that the progress bar is created with the correct total and outputs 'Processing'."""
        from tqdm import tqdm
        
        # Capture tqdm output in an io.StringIO, so we avoid monkey-patching
        mock_stdout = io.StringIO()
        
        # Use a small range but keep total = infinity
        progress_bar = tqdm(range(5), total=sample_324.sol_dict['total'], file=mock_stdout)
        for progress in progress_bar:
            progress_bar.set_description(f"Processing {progress}")
        
        output = mock_stdout.getvalue()
        self.assertIn("Processing", output, "Expected 'Processing' in tqdm's output")
        
        # The progress bar should be created with infinity as total
        self.assertEqual(progress_bar.total, float('inf'))
        progress_bar.close()
    
    def test_progress_bar_description(self):
        """Test that the progress bar description is set (and stored) correctly."""
        from tqdm import tqdm
        
        # Again, capture output but we only need to check the progress_bar.desc
        mock_stdout = io.StringIO()
        test_range = range(5)
        progress_bar = tqdm(test_range, total=len(test_range), file=mock_stdout)
        
        # Set the description for a specific value
        test_value = 3
        progress_bar.set_description(f"Processing {test_value}")
        
        # TQDM typically appends a ": " after the desc, so just check the string starts correctly
        self.assertTrue(progress_bar.desc.startswith(f"Processing {test_value}"),
                        "Progress bar description does not match the expected format.")
        
        progress_bar.close()

if __name__ == '__main__':
    unittest.main()