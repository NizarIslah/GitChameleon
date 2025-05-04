# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tornado.ioloop
from sample_265 import custom_get_ioloop


loop1 = custom_get_ioloop()
loop2 = custom_get_ioloop()
assert loop1 is loop2

loop_current = custom_get_ioloop()
assert loop_current is not None
