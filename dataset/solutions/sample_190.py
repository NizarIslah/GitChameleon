# library: sympy
# version: 1.13
# extra_dependencies: []
from sympy.physics.mechanics import *
import sympy.physics.mechanics


def custom_body(
    rigid_body_text: str, particle_text: str
) -> tuple[sympy.physics.mechanics.RigidBody, sympy.physics.mechanics.Particle]:
    return RigidBody(rigid_body_text), Particle(particle_text)
