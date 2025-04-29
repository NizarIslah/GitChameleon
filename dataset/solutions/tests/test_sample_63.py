import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_63 import combined

class TestCombined(unittest.TestCase):
    def test_combined_function(self):
        # Create test DataFrames
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        
        # Create test Series
        series1 = pd.Series([10, 20])
        series2 = pd.Series([30, 40])
        
        # Call the function
        result_df, result_series = combined(df1, df2, series1, series2)
        
        # Check that the result is a tuple
        self.assertIsInstance((result_df, result_series), tuple)
        
        # Check that the DataFrame was properly appended
        expected_df = pd.DataFrame({'A': [1, 2, 5, 6], 'B': [3, 4, 7, 8]})
        pd.testing.assert_frame_equal(result_df, expected_df)
        
        # Check that the Series was properly appended
        expected_series = pd.Series([10, 20, 30, 40])
        pd.testing.assert_series_equal(result_series, expected_series)
    
    def test_empty_dataframes(self):
        # Test with empty DataFrames
        df1 = pd.DataFrame({'A': [], 'B': []})
        df2 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        series1 = pd.Series([])
        series2 = pd.Series([5, 6])
        
        result_df, result_series = combined(df1, df2, series1, series2)
        
        # Check results
        expected_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd.testing.assert_frame_equal(result_df, expected_df)
        
        expected_series = pd.Series([5, 6])
        pd.testing.assert_series_equal(result_series, expected_series)
    
    def test_different_column_dataframes(self):
        # Test with DataFrames that have different columns
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'C': [7, 8]})
        series1 = pd.Series([10, 20])
        series2 = pd.Series([30, 40])
        
        result_df, result_series = combined(df1, df2, series1, series2)
        
        # The resulting DataFrame should have columns A, B, and C
        self.assertSetEqual(set(result_df.columns), {'A', 'B', 'C'})
        
        # Check that values are preserved
        self.assertEqual(result_df['A'].tolist(), [1, 2, 5, 6])
        self.assertEqual(result_df['B'].iloc[0:2].tolist(), [3, 4])
        self.assertTrue(pd.isna(result_df['B'].iloc[2:]).all())
        self.assertEqual(result_df['C'].iloc[2:].tolist(), [7, 8])
        self.assertTrue(pd.isna(result_df['C'].iloc[0:2]).all())

if __name__ == '__main__':
    unittest.main()