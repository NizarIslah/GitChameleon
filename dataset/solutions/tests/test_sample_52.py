import unittest
import sys
import os

# Add the parent directory to sys.path to import the module to test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_52 import get_scorer_names


class TestSample52(unittest.TestCase):
    def test_get_scorer_names(self):
        """Test that get_scorer_names returns a non-empty list of scorer names."""
        scorer_names = get_scorer_names()

        # Check that the function returns a list
        self.assertIsInstance(scorer_names, list)

        # Check that the list is not empty
        self.assertTrue(len(scorer_names) > 0)

        # Check that all elements in the list are strings
        for name in scorer_names:
            self.assertIsInstance(name, str)

        # Check that common scorers are included
        # These are standard scorers in scikit-learn that should be present
        common_scorers = ["accuracy", "f1", "precision", "recall"]
        for scorer in common_scorers:
            self.assertTrue(
                any(scorer in name for name in scorer_names),
                f"Expected to find a scorer containing '{scorer}' in the list",
            )

        # Additional test specific to sample_52.py implementation
        # Verify that the result matches what we would get directly from SCORERS.keys()
        from sklearn import metrics

        expected_scorers = list(metrics.SCORERS.keys())
        self.assertEqual(
            set(scorer_names),
            set(expected_scorers),
            "The returned scorer names should match metrics.SCORERS.keys()",
        )


if __name__ == "__main__":
    unittest.main()
