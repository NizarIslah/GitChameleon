import os
import importlib.util
import io
import sys
import unittest
from unittest.mock import patch

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sample_324

class TestSample324(unittest.TestCase):

    # Load sample_324 module
    @classmethod
    def setUpClass(cls):
        self.sample_324 = importlib.import_module('sample_324')
        # Ensure the module is loaded correctly
        assert self.sample_324 is not None, "Failed to load sample_324 module"
        # Check if the infinite function exists
        assert hasattr(self.sample_324, 'infinite'), "infinite function not found in sample_324 module"
        # Check if sol_dict is defined
        assert hasattr(self.sample_324, 'sol_dict'), "sol_dict not found in sample_324 module"

    def test_infinite_generator(self):
        """Test that the infinite generator yields values from 0 to 999 and then stops."""
        generator = self.sample_324.infinite()
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
        self.assertEqual(self.sample_324.sol_dict['total'], float('inf'))

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_progress_bar_creation(self, mock_stdout):
        """Test that the progress bar is created with the correct total."""
        from tqdm import tqdm

        # We'll limit the iterations for speed
        with patch.object(self.sample_324, 'infinite', return_value=range(5)):
            progress_bar = tqdm(self.sample_324.infinite(), total=self.sample_324.sol_dict['total'])
            for progress in progress_bar:
                progress_bar.set_description(f"Processing {progress}")

        output = mock_stdout.getvalue()
        self.assertIn("Processing", output)
        self.assertEqual(progress_bar.total, float('inf'))

    def test_progress_bar_description(self):
        """Test that the progress bar description is set correctly."""
        from tqdm import tqdm

        test_range = range(5)
        progress_bar = tqdm(test_range, total=len(test_range))

        test_value = 3
        progress_bar.set_description(f"Processing {test_value}")

        self.assertEqual(progress_bar.desc, f"Processing {test_value}")
        progress_bar.close()

if __name__ == '__main__':
    unittest.main()