import unittest
import sys
import os
import numpy as np

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
import lightgbm as lgb
from sklearn.datasets import make_classification


class TestSample84(unittest.TestCase):
    """Test cases for the LightGBM cross-validation in sample_84.py."""

    def test_lightgbm_dataset_creation(self):
        """Test creating a LightGBM dataset with proper features and labels."""
        # Create a small dataset for testing
        X, y = make_classification(n_samples=100, n_features=5,
                                  n_informative=2, n_redundant=2,
                                  random_state=42)

        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)

        # Verify dataset properties
        self.assertEqual(train_data.num_data(), 100)
        self.assertEqual(train_data.num_feature(), 5)

        # Check if labels are correctly assigned
        retrieved_labels = train_data.get_label()
        np.testing.assert_array_equal(retrieved_labels, y)

    def test_lightgbm_cv_execution(self):
        """Test that LightGBM cross-validation runs without errors."""
        # Create a small dataset for testing
        X, y = make_classification(n_samples=100, n_features=5,
                                  n_informative=2, n_redundant=2,
                                  random_state=42)

        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)

        # Define parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'verbose': -1
        }

        # Run cross-validation
        cv_results = lgb.cv(
            params=params,
            train_set=train_data,
            num_boost_round=10,  # Reduced for faster testing
            nfold=3,
            early_stopping_rounds=5,
            eval_train_metric=True
        )

        # Adapt checks to match keys returned by newer LightGBM versions
        self.assertIn('train binary_logloss-mean', cv_results)
        self.assertIn('train binary_logloss-stdv', cv_results)
        self.assertIn('valid binary_logloss-mean', cv_results)
        self.assertIn('valid binary_logloss-stdv', cv_results)

        # Check that we have results for each iteration
        # (CV can stop early, but it cannot exceed the number of boost rounds)
        self.assertLessEqual(len(cv_results['valid binary_logloss-mean']), 10)

    def test_cv_early_stopping(self):
        """Test that early stopping works in LightGBM cross-validation."""
        # Create a dataset where early stopping is likely to occur
        # Increase n_informative so that n_classes(2)*n_clusters_per_class(2)=4 
        # is ≤ 2^n_informative
        X, y = make_classification(n_samples=100, n_features=5,
                                  n_informative=3, n_redundant=1,
                                  random_state=42)

        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)

        # Define parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.1,
            'verbose': -1
        }

        # Run cross-validation with early stopping
        cv_results = lgb.cv(
            params=params,
            train_set=train_data,
            num_boost_round=50,
            nfold=3,
            early_stopping_rounds=3,
            eval_train_metric=True
        )

        # Verify that early stopping likely occurred (iterations < max)
        self.assertLess(len(cv_results['valid binary_logloss-mean']), 50)

    def test_parameter_effects(self):
        """Test that changing parameters affects the cross-validation results."""
        # Create a dataset
        X, y = make_classification(n_samples=100, n_features=5,
                                  n_informative=2, n_redundant=2,
                                  random_state=42)

        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)

        # Define two sets of parameters with different learning rates
        params_slow = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.01,
            'verbose': -1
        }

        params_fast = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.1,
            'verbose': -1
        }

        # Run cross-validation with both parameter sets
        cv_results_slow = lgb.cv(
            params=params_slow,
            train_set=train_data,
            num_boost_round=10,
            nfold=3,
            early_stopping_rounds=None,
            eval_train_metric=True
        )

        cv_results_fast = lgb.cv(
            params=params_fast,
            train_set=train_data,
            num_boost_round=10,
            nfold=3,
            early_stopping_rounds=None,
            eval_train_metric=True
        )

        # Compare final validation loss values for both sets
        slow_loss_final = cv_results_slow['valid binary_logloss-mean'][-1]
        fast_loss_final = cv_results_fast['valid binary_logloss-mean'][-1]

        # Verify that the faster learning rate leads to a different (usually lower) final loss
        self.assertNotEqual(slow_loss_final, fast_loss_final)


if __name__ == '__main__':
    unittest.main()