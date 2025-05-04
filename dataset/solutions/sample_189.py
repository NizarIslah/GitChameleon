# library: sympy
# version: 1.13
# extra_dependencies: []
from sympy import symbols
from sympy.physics.mechanics import Particle, PinJoint, PrismaticJoint, RigidBody
import sympy
import sympy.physics.mechanics


def custom_motion(
    wall: sympy.physics.mechanics.RigidBody,
    slider: sympy.physics.mechanics.PrismaticJoint,
    pin: sympy.physics.mechanics.PinJoint,
) -> sympy.Matrix:
    from sympy.physics.mechanics import System

    system = System.from_newtonian(wall)
    system.add_joints(slider, pin)
    return system.form_eoms()
