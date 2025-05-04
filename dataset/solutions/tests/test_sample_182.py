import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_182 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_182 import custom_pinJoint_connect
from sympy.physics.mechanics import Body, PinJoint
import sympy as sp


class TestCustomPinJointConnect(unittest.TestCase):
    def setUp(self):
        # Create two Body objects for testing
        self.parent_body = Body("parent")
        self.child_body = Body("child")

    def test_custom_pinJoint_connect(self):
        # Call the function with the test bodies
        pin_joint = custom_pinJoint_connect(self.parent_body, self.child_body)

        # Verify that the returned object is a PinJoint
        self.assertIsInstance(pin_joint, PinJoint)

        # Verify the name of the joint
        self.assertEqual(pin_joint.name, "pin")

        # Verify the parent and child bodies
        self.assertEqual(pin_joint.parent, self.parent_body)
        self.assertEqual(pin_joint.child, self.child_body)

        # Verify the connection points - note that PinJoint creates Point objects
        # with specific names rather than using the vectors directly
        self.assertEqual(pin_joint.parent_point.name, "pin_parent_joint")
        self.assertEqual(pin_joint.child_point.name, "pin_child_joint")

        # Verify the types of the connection points
        from sympy.physics.vector.point import Point

        self.assertIsInstance(pin_joint.parent_point, Point)
        self.assertIsInstance(pin_joint.child_point, Point)


if __name__ == "__main__":
    unittest.main()
