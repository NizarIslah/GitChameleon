#!/usr/bin/env python3
# Test file for sample_323.py

import io
import sys
import unittest
from unittest.mock import patch

from tqdm import tqdm

sys.path.append("/repo/dataset/solutions")
from sample_323 import infinite, sol_dict

assert sol_dict["total"] is None
