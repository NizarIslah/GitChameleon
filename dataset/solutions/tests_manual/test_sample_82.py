import sys
import os
import unittest
import numpy as np
import pytest
import lightgbm as lgb

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sample_82

import numpy as np

assert "cvbooster" in cv_results
assert len(cv_results["cvbooster"].boosters) == NFOLD
assert all(
    isinstance(booster, lgb.Booster) for booster in cv_results["cvbooster"].boosters
)
