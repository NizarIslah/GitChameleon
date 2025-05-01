import os
import importlib.util
import io
import sys
import unittest
from unittest.mock import patch

# Resolve the path to sample_324.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
SAMPLE_324_PATH = os.path.join(PARENT_DIR, "sample_324.py")

# Check if sample_324.py actually exists
def sample_module_available():
    return os.path.isfile(SAMPLE_324_PATH)

# Only define and run tests if sample_324.py is found
@unittest.skipUnless(sample_module_available(), "sample_324.py not found, skipping these tests.")
class TestSample324(unittest.TestCase):

    # Load sample_324 module
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("sample_324", SAMPLE_324_PATH)
        cls.sample_324 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.sample_324)

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