import importlib.util
import os
import unittest

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

if __name__ == '__main__':
    unittest.main()
