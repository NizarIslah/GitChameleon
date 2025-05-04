import os
import sys
import unittest

from sympy.physics.mechanics import Body, PinJoint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_181 import custom_pinJoint


class TestCustomPinJoint(unittest.TestCase):
    def test_custom_pinJoint(self):
        # Create parent and child bodies
        parent = Body("parent")
        child = Body("child")

        # Call the function to create a pin joint
        pin_joint = custom_pinJoint(parent, child)

        # Verify the pin joint was created correctly
        self.assertIsInstance(pin_joint, PinJoint)
        self.assertEqual(pin_joint.name, "pin")
        self.assertEqual(pin_joint.parent, parent)
        self.assertEqual(pin_joint.child, child)

    def test_with_different_bodies(self):
        # Test with different body names
        body1 = Body("body1")
        body2 = Body("body2")

        pin_joint = custom_pinJoint(body1, body2)

        # Verify basic properties
        self.assertIsInstance(pin_joint, PinJoint)
        self.assertEqual(pin_joint.name, "pin")
        self.assertEqual(pin_joint.parent, body1)
        self.assertEqual(pin_joint.child, body2)


if __name__ == "__main__":
    unittest.main()
