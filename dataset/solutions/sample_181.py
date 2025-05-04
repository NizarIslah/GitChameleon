# library: sympy
# version: 1.12
# extra_dependencies: []
from sympy.physics.mechanics import Body, PinJoint
import sympy.physics.mechanics


def custom_pinJoint(
    parent: sympy.physics.mechanics.Body, child: sympy.physics.mechanics.Body
) -> sympy.physics.mechanics.PinJoint:
    return PinJoint(
        "pin", parent, child, parent_point=parent.frame.x, child_point=-child.frame.x
    )
