import pytest
import torch
import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scipy.stats import norm
from sample_0 import log_ndtr


class TestLogNdtr:
    @pytest.fixture
    def setup_tensors(self):
        # Fixture to set up common tensors for testing
        self.float_tensor = torch.tensor(
            [0.0, 1.0, -1.0, 2.0, -2.0], dtype=torch.float32
        )
        self.double_tensor = torch.tensor(
            [0.0, 1.0, -1.0, 2.0, -2.0], dtype=torch.float64
        )
        self.large_tensor = torch.tensor(
            [10.0, -10.0, 20.0, -20.0], dtype=torch.float32
        )
        self.empty_tensor = torch.tensor([], dtype=torch.float32)
        self.single_value_tensor = torch.tensor([0.0], dtype=torch.float32)
        self.nonfinite_tensor = torch.tensor(
            [float("inf"), float("-inf"), float("nan")], dtype=torch.float32
        )

    def test_basic_functionality_with_float_tensor(self, setup_tensors):
        """Test log_ndtr with a float tensor to ensure correct log CDF values are returned."""
        result = log_ndtr(self.float_tensor)
        # Calculate expected values using scipy's norm.logcdf directly
        expected = (
            torch.from_numpy(norm.logcdf(self.float_tensor.numpy()))
            .to(dtype=torch.float32)
            .to(dtype=result.dtype, device=result.device)
        )
        assert torch.allclose(result, expected, atol=1e-4)

    def test_double_precision_tensor_handling(self, setup_tensors):
        """Test log_ndtr with a double tensor to ensure correct log CDF values are returned."""
        result = log_ndtr(self.double_tensor)
        expected = torch.from_numpy(norm.logcdf(self.double_tensor.numpy()))

        assert torch.allclose(
            result, expected, atol=1e-8
        )  # Tighter tolerance for double precision
        assert result.dtype == torch.float64  # Check that dtype is preserved

    def test_large_positive_negative_values(self, setup_tensors):
        """Test log_ndtr with large positive and negative values to ensure stability."""
        result = log_ndtr(self.large_tensor)
        expected = torch.from_numpy(norm.logcdf(self.large_tensor.numpy())).to(
            dtype=result.dtype, device=result.device
        )

        # For very large values, we expect log(1) ≈ 0 for large positive values
        # and log values approaching -inf for large negative values
        assert torch.allclose(result, expected, atol=1e-4, equal_nan=True)

    def test_empty_tensor_handling(self, setup_tensors):
        """Test log_ndtr with an empty tensor to ensure it returns an empty tensor."""
        result = log_ndtr(self.empty_tensor)

        assert result.numel() == 0  # Check that result is also empty

    def test_non_finite_values(self, setup_tensors):
        """Test log_ndtr with non-finite values (inf, -inf, nan) to ensure proper handling."""
        result = log_ndtr(self.nonfinite_tensor)
        zero = torch.tensor(0.0, dtype=result.dtype, device=result.device)
        neg_inf = torch.tensor(-float("inf"), dtype=result.dtype, device=result.device)
        assert torch.isclose(result[0], zero, atol=1e-4)  # log(1) for +inf
        assert torch.isclose(result[1], neg_inf, atol=1e-4, equal_nan=False)
        assert torch.isnan(result[2])  # nan for nan

    def test_single_value_tensor(self, setup_tensors):
        """Test log_ndtr with a single value tensor to ensure it handles single-element tensors."""
        result = log_ndtr(self.single_value_tensor)
        expected = torch.from_numpy(norm.logcdf(self.single_value_tensor.numpy())).to(
            dtype=result.dtype, device=result.device
        )
        assert torch.allclose(result, expected, atol=1e-4)

    # def test_preserves_input_tensor_dtype(self, setup_tensors):
    #     """Test that log_ndtr preserves the input tensor's dtype in the output."""
    #     # Test with float32
    #     float_result = log_ndtr(self.float_tensor)
    #     assert float_result.dtype == torch.float32

    #     # Test with float64
    #     double_result = log_ndtr(self.double_tensor)
    #     assert double_result.dtype == torch.float64

    #     # Test with a different dtype if supported
    #     if torch.cuda.is_available():
    #         half_tensor = self.float_tensor.half()  # Convert to float16
    #         half_result = log_ndtr(half_tensor)
    #         assert half_result.dtype == torch.float16
