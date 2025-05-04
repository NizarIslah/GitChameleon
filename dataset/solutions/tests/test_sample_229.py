import pytest
import pathlib
import sys
from unittest.mock import MagicMock

# Add the parent directory to sys.path to import the module
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import the module to test
import sample_229


class TestPytestCollectFile:
    def test_pytest_collect_file_exists(self):
        """Test that the pytest_collect_file hook exists."""
        assert hasattr(sample_229, "pytest_collect_file")
        assert callable(sample_229.pytest_collect_file)

    def test_pytest_collect_file_accepts_path_parameter(self):
        """Test that the pytest_collect_file hook accepts a Path parameter."""
        # Create a mock Path object
        mock_path = MagicMock(spec=pathlib.Path)

        # Call the function with the mock path
        # This should not raise any exceptions if the parameter type is correct
        sample_229.pytest_collect_file(mock_path)

    def test_pytest_collect_file_returns_none(self):
        """Test that the pytest_collect_file hook returns None (default behavior)."""
        mock_path = MagicMock(spec=pathlib.Path)
        result = sample_229.pytest_collect_file(mock_path)
        assert result is None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
