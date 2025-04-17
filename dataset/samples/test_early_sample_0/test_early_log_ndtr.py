import pytest
import torch
import numpy as np
from sample_0 import log_ndtr

@pytest.mark.usefixtures("setup_torch")
class TestLogNdtr:
    @pytest.fixture(scope="class")
    def setup_torch(self):
        # Setup code if needed, for now, it's just a placeholder
        pass

    @pytest.mark.happy_path
    def test_log_ndtr_positive_values(self):
        """Test log_ndtr with positive input values."""
        input_tensor = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        expected_output = torch.from_numpy(np.log(np.array([0.691462, 0.841345, 0.977250])))
        output = log_ndtr(input_tensor)
        assert torch.allclose(output, expected_output, atol=1e-5), f"Expected {expected_output}, but got {output}"

    @pytest.mark.happy_path
    def test_log_ndtr_negative_values(self):
        """Test log_ndtr with negative input values."""
        input_tensor = torch.tensor([-0.5, -1.0, -2.0], dtype=torch.float32)
        expected_output = torch.from_numpy(np.log(np.array([0.308538, 0.158655, 0.022750])))
        output = log_ndtr(input_tensor)
        assert torch.allclose(output, expected_output, atol=1e-5), f"Expected {expected_output}, but got {output}"

    @pytest.mark.happy_path
    def test_log_ndtr_zero(self):
        """Test log_ndtr with zero as input."""
        input_tensor = torch.tensor([0.0], dtype=torch.float32)
        expected_output = torch.from_numpy(np.log(np.array([0.5])))
        output = log_ndtr(input_tensor)
        assert torch.allclose(output, expected_output, atol=1e-5), f"Expected {expected_output}, but got {output}"

    # @pytest.mark.edge_case
    # def test_log_ndtr_large_values(self):
    #     """Test log_ndtr with large positive and negative values."""
    #     input_tensor = torch.tensor([10.0, -10.0], dtype=torch.float32)
    #     expected_output = torch.tensor([0.0, float('-inf')], dtype=torch.float64)
    #     output = log_ndtr(input_tensor)
    #     assert torch.allclose(output, expected_output, atol=1e-5), f"Expected {expected_output}, but got {output}"

    @pytest.mark.edge_case
    def test_log_ndtr_empty_tensor(self):
        """Test log_ndtr with an empty tensor."""
        input_tensor = torch.tensor([], dtype=torch.float32)
        expected_output = torch.tensor([], dtype=torch.float64)
        output = log_ndtr(input_tensor)
        assert torch.equal(output, expected_output), f"Expected {expected_output}, but got {output}"

    @pytest.mark.edge_case
    def test_log_ndtr_non_finite_values(self):
        """Test log_ndtr with non-finite values (NaN, Inf)."""
        input_tensor = torch.tensor([float('nan'), float('inf'), float('-inf')], dtype=torch.float32)
        output = log_ndtr(input_tensor)
        assert torch.isnan(output[0]), f"Expected NaN, but got {output[0]}"
        assert output[1] == 0.0, f"Expected 0.0 for positive infinity, but got {output[1]}"
        assert output[2] == float('-inf'), f"Expected -inf for negative infinity, but got {output[2]}"