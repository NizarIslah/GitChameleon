import importlib.util
import io
import sys
import unittest
import os

from tqdm import tqdm

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import the module from the solutions directory
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
        """Test that the progress bar is created with the correct total."""
        output_capture = io.StringIO()

        # Use the real generator and limit iterations to keep the test fast
        infinite_gen = sample_324.infinite()
        progress_bar = tqdm(infinite_gen, total=sample_324.sol_dict['total'], file=output_capture)
        
        # Only iterate a few times
        count = 0
        for progress in progress_bar:
            progress_bar.set_description(f"Processing {progress}")
            count += 1
            if count >= 5:
                break

        # Check that the output contains the expected description
        output = output_capture.getvalue()
        self.assertIn("Processing", output)
        
        # The progress bar should have infinity as the total
        self.assertEqual(progress_bar.total, float('inf'))
        progress_bar.close()
    
    def test_progress_bar_description(self):
        """Test that the progress bar description is set correctly."""
        output_capture = io.StringIO()

        # Create a progress bar with a small range
        test_range = range(5)
        progress_bar = tqdm(test_range, total=len(test_range), file=output_capture)
        
        # Set the description for a specific value
        test_value = 3
        progress_bar.set_description(f"Processing {test_value}")
        
        # Fix: remove trailing colon and space for comparison
        self.assertEqual(progress_bar.desc.rstrip(': '), f"Processing {test_value}")
        
        progress_bar.close()

if __name__ == '__main__':
    unittest.main()