# library: sympy
# version: 1.13
# extra_dependencies: []
from sympy import symbols
from sympy.physics.mechanics import ReferenceFrame
import sympy.physics.vector


def custom_generateInertia(
    N: sympy.physics.vector.frame.ReferenceFrame,
    Ixx: sympy.Symbol,
    Iyy: sympy.Symbol,
    Izz: sympy.Symbol,
) -> sympy.physics.vector.dyadic.Dyadic:
    from sympy.physics.mechanics import inertia

    return inertia(N, Ixx, Iyy, Izz)
