# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_272 import custom_chart_studio_usage


class TestSample272(unittest.TestCase):
    def test_custom_chart_studio_usage(self):
        """Test that custom_chart_studio_usage correctly identifies the plot attribute."""
        # The function should return True if chart_studio.plotly has a plot attribute
        result = custom_chart_studio_usage()

        # Since we're using the specified dependencies (plotly 4.0.0 and chart-studio 1.0.0),
        # we expect the plot attribute to exist in chart_studio.plotly
        self.assertTrue(result, "chart_studio.plotly should have a 'plot' attribute")

        # Additional verification - directly check for the attribute
        import chart_studio.plotly


import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    has_plot = custom_chart_studio_usage()
    for warn in w:
        assert not issubclass(warn.category, DeprecationWarning), "Deprecated API used!"

assert has_plot
