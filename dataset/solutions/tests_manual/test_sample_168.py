# Add the parent directory to import sys
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import werkzeug.exceptions

sys.path.append(str(Path(__file__).parent.parent))
from sample_168 import error404, stack_and_save


base_path = "/var/www/myapp"
sub_path = "../secret.txt"


a = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
b = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float64)
arr_list = [a, b]
casting_policy = "safe"
out_dtype = np.float64
stacked_correct = np.vstack(arr_list).astype(np.float64)
try:
    joined, stacked = stack_and_save(
        arr_list, base_path, sub_path, casting_policy, out_dtype
    )
except werkzeug.exceptions.NotFound as e:
    assertion_result = True
else:
    assertion_result = False
assert assertion_result

base_path = "/var/www/myapp"
sub_path = "secret.txt"
joined, stacked = stack_and_save(
    arr_list, base_path, sub_path, casting_policy, out_dtype
)
assertion_result = (
    joined == "/var/www/myapp/secret.txt"
    and np.array_equal(stacked, stacked_correct)
    and stacked.dtype == np.float64
)
assert assertion_result

stacked_correct = stacked_correct.astype(np.float32)
out_dtype = np.float32
try:
    joined, stacked = stack_and_save(
        arr_list, base_path, sub_path, casting_policy, out_dtype
    )
except TypeError as e:
    assertion_result = True
else:
    assertion_result = False
assert assertion_result

stacked_correct = stacked_correct.astype(np.float32)
out_dtype = np.float32
casting_policy = "unsafe"
joined, stacked = stack_and_save(
    arr_list, base_path, sub_path, casting_policy, out_dtype
)

assertion_result = (
    joined == "/var/www/myapp/secret.txt"
    and np.array_equal(stacked, stacked_correct)
    and stacked.dtype == np.float32
)
assert assertion_result
