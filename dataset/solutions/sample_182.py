# library: sympy
# version: 1.12
# extra_dependencies: []
from sympy.physics.mechanics import Body, PinJoint
import sympy.physics.mechanics
import sympy as sp


def custom_pinJoint_connect(
    parent: sympy.physics.mechanics.Body, child: sympy.physics.mechanics.Body
) -> sympy.physics.mechanics.PinJoint:
    return PinJoint(
        "pin", parent, child, parent_point=parent.frame.x, child_point=-child.frame.x
    )
