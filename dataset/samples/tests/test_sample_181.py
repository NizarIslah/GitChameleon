import unittest
from sympy.physics.mechanics import Body, PinJoint
from dataset.solutions.sample_181 import custom_pinJoint

class TestCustomPinJoint(unittest.TestCase):
    def test_custom_pinJoint(self):
        # Create parent and child bodies
        parent = Body('parent')
        child = Body('child')
        
        # Call the function to create a pin joint
        pin_joint = custom_pinJoint(parent, child)
        
        # Verify the pin joint was created correctly
        self.assertIsInstance(pin_joint, PinJoint)
        self.assertEqual(pin_joint.name, 'pin')
        self.assertEqual(pin_joint.parent, parent)
        self.assertEqual(pin_joint.child, child)
        
        # Verify the connection points
        self.assertEqual(pin_joint.parent_point, parent.frame.x)
        self.assertEqual(pin_joint.child_point, -child.frame.x)
    
    def test_with_different_bodies(self):
        # Test with different body names
        body1 = Body('body1')
        body2 = Body('body2')
        
        pin_joint = custom_pinJoint(body1, body2)
        
        # Verify basic properties
        self.assertIsInstance(pin_joint, PinJoint)
        self.assertEqual(pin_joint.name, 'pin')
        self.assertEqual(pin_joint.parent, body1)
        self.assertEqual(pin_joint.child, body2)
        
        # Verify connection points
        self.assertEqual(pin_joint.parent_point, body1.frame.x)
        self.assertEqual(pin_joint.child_point, -body2.frame.x)

if __name__ == '__main__':
    unittest.main()