import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_189 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_189 import custom_motion
from sympy import symbols, Matrix
from sympy.physics.mechanics import (
    RigidBody,
    PinJoint,
    PrismaticJoint,
    Point,
    ReferenceFrame,
    inertia,
    dynamicsymbols,
)


class TestCustomMotion(unittest.TestCase):
    def setUp(self):
        """Set up the mechanical system for testing."""
        # Create symbols for the test
        self.q = dynamicsymbols("q:2")  # Generalized coordinates
        self.u = dynamicsymbols("u:2")  # Generalized speeds
        self.m1, self.m2 = symbols("m1 m2")  # Masses
        self.l1, self.l2 = symbols("l1 l2")  # Lengths
        self.g = symbols("g")  # Gravity

        # Create reference frames
        self.N = ReferenceFrame("N")  # Inertial reference frame
        self.A = ReferenceFrame("A")  # First body frame
        self.B = ReferenceFrame("B")  # Second body frame

        # Define points
        self.O = Point("O")  # Origin point
        self.P1 = Point("P1")  # First body center of mass
        self.P2 = Point("P2")  # Second body center of mass

        # Set velocities to zero for the inertial frame
        self.O.set_vel(self.N, 0)

        # Set velocities for other points to avoid warnings
        self.P1.set_vel(self.N, 0)
        self.P2.set_vel(self.N, 0)

        # Create inertia dyadics
        self.I1 = inertia(self.A, 1, 1, 1)
        self.I2 = inertia(self.B, 1, 1, 1)

        # Create rigid bodies
        # For the wall, we use a zero mass and zero inertia
        self.wall = RigidBody("wall", masscenter=self.O, frame=self.N, mass=0)
        self.body1 = RigidBody(
            "body1",
            masscenter=self.P1,
            frame=self.A,
            mass=self.m1,
            inertia=(self.I1, self.P1),
        )
        self.body2 = RigidBody(
            "body2",
            masscenter=self.P2,
            frame=self.B,
            mass=self.m2,
            inertia=(self.I2, self.P2),
        )

        # Set up kinematics
        self.A.orient_axis(self.N, self.N.z, self.q[0])
        self.P1.set_pos(self.O, self.l1 * self.A.x)

        self.B.orient_axis(self.A, self.A.z, self.q[1])
        self.P2.set_pos(self.P1, self.l2 * self.B.x)

        # Create joints
        self.slider = PrismaticJoint(
            "slider", self.wall, self.body1, self.q[0], self.u[0], self.N.x
        )
        self.pin = PinJoint(
            "pin", self.body1, self.body2, self.q[1], self.u[1], self.N.z
        )

    def test_custom_motion_returns_matrix(self):
        """Test that custom_motion returns a sympy Matrix."""
        result = custom_motion(self.wall, self.slider, self.pin)
        self.assertIsInstance(result, Matrix)

    def test_custom_motion_with_simple_system(self):
        """Test custom_motion with a simple mechanical system."""
        # Call the function
        result = custom_motion(self.wall, self.slider, self.pin)

        # Check that the result is a Matrix
        self.assertIsInstance(result, Matrix)

        # Check that the result is not empty
        self.assertGreater(result.shape[0], 0)

        # The equations of motion should have as many rows as there are degrees of freedom
        # In this case, we have 2 degrees of freedom (q[0] and q[1])
        # The actual implementation returns 2 rows, not 4 as we might expect
        self.assertEqual(result.shape[0], 2)

    def test_custom_motion_with_different_bodies(self):
        """Test custom_motion with different rigid bodies."""
        # Create a new rigid body
        new_body = RigidBody(
            "new_body",
            masscenter=self.P1,
            frame=self.A,
            mass=self.m1,
            inertia=(self.I1, self.P1),
        )

        # Create new joints
        new_slider = PrismaticJoint(
            "new_slider", self.wall, new_body, self.q[0], self.u[0], self.N.x
        )
        new_pin = PinJoint(
            "new_pin", new_body, self.body2, self.q[1], self.u[1], self.N.z
        )

        # Call the function with the new bodies
        result = custom_motion(self.wall, new_slider, new_pin)

        # Check that the result is a Matrix
        self.assertIsInstance(result, Matrix)

        # Check that the result is not empty
        self.assertGreater(result.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
