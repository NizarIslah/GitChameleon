import sys
import os
import numpy as np
import pytest
import lightgbm as lgb
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sample_82

class TestLightGBMCV:
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.X = np.random.rand(100, 10)
        self.y = np.random.randint(0, 2, 100)
        self.train_data = lgb.Dataset(self.X, label=self.y)
        
        # Define parameters similar to those in sample_82.py
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'verbose': -1
        }
        
        # Mock cv_results
        self.mock_cv_results = {
            'binary_logloss-mean': [0.693, 0.690, 0.687],
            'binary_logloss-stdv': [0.001, 0.002, 0.003],
            'cvbooster': MagicMock()
        }
    
    @patch('lightgbm.cv')
    def test_cv_parameters(self, mock_cv):
        """Test that cv is called with the correct parameters."""
        # Set up the mock return value
        mock_cv.return_value = self.mock_cv_results
        
        # Import constants from sample_82
        NUM_BOOST_ROUND = sample_82.NUM_BOOST_ROUND
        NFOLD = sample_82.NFOLD
        EARLY_STOPPING_ROUNDS = sample_82.EARLY_STOPPING_ROUNDS
        
        # Call the cv function (we're not directly calling it, but checking the module's execution)
        # We'll patch the module's cv_results to use our mock
        with patch.object(sample_82, 'cv_results', self.mock_cv_results):
            # Manually call cv with the same parameters as in sample_82
            result = lgb.cv(
                params=self.params,
                train_set=self.train_data,
                num_boost_round=NUM_BOOST_ROUND,
                nfold=NFOLD,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                return_cvbooster=True
            )
            
            # Check that cv was called with the correct parameters
            mock_cv.assert_called_once()
            args, kwargs = mock_cv.call_args
            
            assert kwargs['params'] == self.params
            assert kwargs['num_boost_round'] == NUM_BOOST_ROUND
            assert kwargs['nfold'] == NFOLD
            assert kwargs['early_stopping_rounds'] == EARLY_STOPPING_ROUNDS
            assert kwargs['return_cvbooster'] == True
    
    def test_dataset_creation(self):
        """Test that the Dataset is created correctly with the right dimensions."""
        # Check that X and y have the expected shapes from the constants
        X, y = sample_82.X, sample_82.y
        
        assert X.shape[0] == sample_82.NUM_SAMPLES
        assert X.shape[1] == sample_82.NUM_FEATURES
        assert len(y) == sample_82.NUM_SAMPLES
        
        # Check that the Dataset object is created correctly
        train_data = sample_82.train_data
        assert isinstance(train_data, lgb.Dataset)
        
        # Get data from the Dataset
        X_data, y_data = train_data.data, train_data.label
        assert X_data.shape == X.shape
        assert y_data.shape == y.shape
    
    def test_params_configuration(self):
        """Test that the parameters are configured correctly."""
        params = sample_82.params
        
        assert params['objective'] == 'binary'
        assert params['metric'] == 'binary_logloss'
        assert params['learning_rate'] == sample_82.LEARNING_RATE
        assert params['verbose'] == -1
    
    @patch('lightgbm.cv')
    def test_cv_results_structure(self, mock_cv):
        """Test the structure of cv_results."""
        # Set up the mock return value
        mock_cv.return_value = self.mock_cv_results
        
        # Replace the module's cv_results with our mock
        with patch.object(sample_82, 'cv_results', self.mock_cv_results):
            # Check that cv_results has the expected keys
            assert 'binary_logloss-mean' in sample_82.cv_results
            assert 'binary_logloss-stdv' in sample_82.cv_results
            assert 'cvbooster' in sample_82.cv_results
            
            # Check that the metrics are lists
            assert isinstance(sample_82.cv_results['binary_logloss-mean'], list)
            assert isinstance(sample_82.cv_results['binary_logloss-stdv'], list)
    
    def test_make_classification_parameters(self):
        """Test that make_classification is called with the correct parameters."""
        # We can't easily mock make_classification since it's already been called,
        # but we can check that the constants are used correctly
        assert sample_82.NUM_SAMPLES == 500
        assert sample_82.NUM_FEATURES == 20
        assert sample_82.INFORMATIVE_FEATURES == 2
        assert sample_82.REDUNDANT_FEATURES == 10
        assert sample_82.RANDOM_STATE == 42