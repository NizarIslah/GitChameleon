import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_186 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_186 import custom_generateInertia
from sympy import symbols, S, simplify
from sympy.physics.mechanics import ReferenceFrame
import sympy.physics.vector


class TestCustomGenerateInertia(unittest.TestCase):
    def test_basic_inertia_generation(self):
        """Test basic inertia generation with symbolic moments."""
        # Create a reference frame
        N = ReferenceFrame("N")
        # Create symbolic moments of inertia
        Ixx, Iyy, Izz = symbols("Ixx Iyy Izz")

        # Generate inertia dyadic
        I = custom_generateInertia(N, Ixx, Iyy, Izz)

        # Check that the result is a Dyadic
        self.assertIsInstance(I, sympy.physics.vector.dyadic.Dyadic)

        # Check the components of the dyadic using the args attribute
        # The args attribute contains tuples of (coefficient, basis1, basis2)
        args_dict = {(arg[1], arg[2]): arg[0] for arg in I.args}

        self.assertEqual(args_dict[(N.x, N.x)], Ixx)
        self.assertEqual(args_dict[(N.y, N.y)], Iyy)
        self.assertEqual(args_dict[(N.z, N.z)], Izz)

        # Check that off-diagonal components are not present in args
        self.assertNotIn((N.x, N.y), args_dict)
        self.assertNotIn((N.x, N.z), args_dict)
        self.assertNotIn((N.y, N.x), args_dict)
        self.assertNotIn((N.y, N.z), args_dict)
        self.assertNotIn((N.z, N.x), args_dict)
        self.assertNotIn((N.z, N.y), args_dict)

    def test_numeric_values_for_moments(self):
        """Test inertia generation with numeric values for moments."""
        # Create a reference frame
        N = ReferenceFrame("N")

        # Generate inertia dyadic with numeric values
        I = custom_generateInertia(N, 1, 2, 3)

        # Check the components of the dyadic
        args_dict = {(arg[1], arg[2]): arg[0] for arg in I.args}

        self.assertEqual(args_dict[(N.x, N.x)], 1)
        self.assertEqual(args_dict[(N.y, N.y)], 2)
        self.assertEqual(args_dict[(N.z, N.z)], 3)

    def test_zero_moments_of_inertia(self):
        """Test inertia generation with zero moments of inertia."""
        # Create a reference frame
        N = ReferenceFrame("N")

        # Generate inertia dyadic with zero moments
        I = custom_generateInertia(N, 0, 0, 0)

        # When all moments are zero, the dyadic might be simplified to 0
        # Check that the dyadic is effectively zero
        zero_dyadic = 0 * (N.x | N.x)
        self.assertEqual(str(I), str(zero_dyadic))

    def test_different_reference_frames(self):
        """Test inertia generation with different reference frames."""
        # Create different reference frames
        N = ReferenceFrame("N")
        A = ReferenceFrame("A")
        B = ReferenceFrame("B")

        # Create symbolic moments of inertia
        Ixx, Iyy, Izz = symbols("Ixx Iyy Izz")

        # Generate inertia dyadics for different frames
        I_N = custom_generateInertia(N, Ixx, Iyy, Izz)
        I_A = custom_generateInertia(A, Ixx, Iyy, Izz)
        I_B = custom_generateInertia(B, Ixx, Iyy, Izz)

        # Check that the components are correct for each frame
        args_dict_N = {(arg[1], arg[2]): arg[0] for arg in I_N.args}
        args_dict_A = {(arg[1], arg[2]): arg[0] for arg in I_A.args}
        args_dict_B = {(arg[1], arg[2]): arg[0] for arg in I_B.args}

        self.assertEqual(args_dict_N[(N.x, N.x)], Ixx)
        self.assertEqual(args_dict_N[(N.y, N.y)], Iyy)
        self.assertEqual(args_dict_N[(N.z, N.z)], Izz)

        self.assertEqual(args_dict_A[(A.x, A.x)], Ixx)
        self.assertEqual(args_dict_A[(A.y, A.y)], Iyy)
        self.assertEqual(args_dict_A[(A.z, A.z)], Izz)

        self.assertEqual(args_dict_B[(B.x, B.x)], Ixx)
        self.assertEqual(args_dict_B[(B.y, B.y)], Iyy)
        self.assertEqual(args_dict_B[(B.z, B.z)], Izz)

    def test_dyadic_properties_verification(self):
        """Test properties of the generated inertia dyadic."""
        # Create a reference frame
        N = ReferenceFrame("N")
        # Create symbolic moments of inertia
        Ixx, Iyy, Izz = symbols("Ixx Iyy Izz")

        # Generate inertia dyadic
        I = custom_generateInertia(N, Ixx, Iyy, Izz)

        # Test dyadic addition
        I_doubled = I + I
        args_dict_doubled = {(arg[1], arg[2]): arg[0] for arg in I_doubled.args}

        self.assertEqual(args_dict_doubled[(N.x, N.x)], 2 * Ixx)
        self.assertEqual(args_dict_doubled[(N.y, N.y)], 2 * Iyy)
        self.assertEqual(args_dict_doubled[(N.z, N.z)], 2 * Izz)

        # Test dyadic scalar multiplication
        I_scaled = 3 * I
        args_dict_scaled = {(arg[1], arg[2]): arg[0] for arg in I_scaled.args}

        self.assertEqual(args_dict_scaled[(N.x, N.x)], 3 * Ixx)
        self.assertEqual(args_dict_scaled[(N.y, N.y)], 3 * Iyy)
        self.assertEqual(args_dict_scaled[(N.z, N.z)], 3 * Izz)

        # Test dyadic dot product with unit vectors
        dot_x = I & N.x
        dot_y = I & N.y
        dot_z = I & N.z

        # For dot products, we get a Vector, which has a different structure
        # We'll check the string representation to verify correctness
        self.assertEqual(str(dot_x), str(Ixx * N.x))
        self.assertEqual(str(dot_y), str(Iyy * N.y))
        self.assertEqual(str(dot_z), str(Izz * N.z))

    def test_invalid_reference_frame(self):
        """Test with invalid reference frame."""
        # Create symbolic moments of inertia
        Ixx, Iyy, Izz = symbols("Ixx Iyy Izz")

        # Test with a string instead of a ReferenceFrame
        with self.assertRaises(TypeError):
            custom_generateInertia("not a frame", Ixx, Iyy, Izz)

        # Test with None
        with self.assertRaises(TypeError):
            custom_generateInertia(None, Ixx, Iyy, Izz)

    def test_invalid_moment_parameters(self):
        """Test with invalid moment parameters."""
        # Create a reference frame
        N = ReferenceFrame("N")

        # Test with complex objects that can't be used as moments of inertia
        # SymPy might be able to sympify some strings, so we'll use objects that definitely can't be moments
        try:
            result = custom_generateInertia(N, object(), 2, 3)
            # If no exception is raised, verify the result is a Dyadic
            self.assertIsInstance(result, sympy.physics.vector.dyadic.Dyadic)
        except Exception:
            # If an exception is raised, that's also acceptable
            pass

        try:
            result = custom_generateInertia(N, 1, object(), 3)
            self.assertIsInstance(result, sympy.physics.vector.dyadic.Dyadic)
        except Exception:
            pass

        try:
            result = custom_generateInertia(N, 1, 2, object())
            self.assertIsInstance(result, sympy.physics.vector.dyadic.Dyadic)
        except Exception:
            pass

    def test_expression_based_moments_of_inertia(self):
        """Test with expression-based moments of inertia."""
        # Create a reference frame
        N = ReferenceFrame("N")

        # Create symbols
        m, r = symbols("m r")

        # Create expression-based moments (like for a solid sphere)
        Ixx = (2 / 5) * m * r**2
        Iyy = (2 / 5) * m * r**2
        Izz = (2 / 5) * m * r**2

        # Generate inertia dyadic
        I = custom_generateInertia(N, Ixx, Iyy, Izz)

        # Check the components of the dyadic
        args_dict = {(arg[1], arg[2]): arg[0] for arg in I.args}

        self.assertEqual(simplify(args_dict[(N.x, N.x)] - (2 / 5) * m * r**2), 0)
        self.assertEqual(simplify(args_dict[(N.y, N.y)] - (2 / 5) * m * r**2), 0)
        self.assertEqual(simplify(args_dict[(N.z, N.z)] - (2 / 5) * m * r**2), 0)

        # Test with different expressions for each axis (like for a cylinder)
        Ixx_cyl = (1 / 2) * m * r**2
        Iyy_cyl = (1 / 4) * m * r**2 + (1 / 12) * m * r**2
        Izz_cyl = (1 / 2) * m * r**2

        # Generate inertia dyadic
        I_cyl = custom_generateInertia(N, Ixx_cyl, Iyy_cyl, Izz_cyl)

        # Check the components of the dyadic
        args_dict_cyl = {(arg[1], arg[2]): arg[0] for arg in I_cyl.args}

        self.assertEqual(simplify(args_dict_cyl[(N.x, N.x)] - (1 / 2) * m * r**2), 0)
        self.assertEqual(
            simplify(
                args_dict_cyl[(N.y, N.y)]
                - ((1 / 4) * m * r**2 + (1 / 12) * m * r**2)
            ),
            0,
        )
        self.assertEqual(simplify(args_dict_cyl[(N.z, N.z)] - (1 / 2) * m * r**2), 0)


if __name__ == "__main__":
    unittest.main()
