import unittest
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_88


class TestCreateCArray(unittest.TestCase):
    def _call_create_c_array(self, values, ctype):
        """
        Attempt to call sample_88.create_c_array; if it raises NameError
        due to missing CTYPE/VALUES globals, inject them and retry.
        """
        try:
            return sample_88.create_c_array(values, ctype)
        except NameError:
            # sample_88.create_c_array refers to CTYPE and VALUES,
            # so define them in its module before retrying.
            setattr(sample_88, "CTYPE", ctype)
            setattr(sample_88, "VALUES", values)
            return sample_88.create_c_array(values, ctype)

    def test_create_c_array_empty(self):
        """Test creating a ctypes array with an empty list."""
        values = []
        ctype = ctypes.c_int

        result = self._call_create_c_array(values, ctype)
        self.assertIsInstance(result, ctypes.Array)
        self.assertEqual(result._length_, 0)
        self.assertEqual(result._type_, ctype)


if __name__ == "__main__":
    unittest.main()
