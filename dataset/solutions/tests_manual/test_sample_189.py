import unittest
import sys
import os

# Add the parent directory to the path so we can import the sample_189 module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_189 import custom_motion
from sympy import symbols, Matrix
from sympy.physics.mechanics import (

l = symbols("l")
wall = RigidBody("wall")
cart = RigidBody("cart")
pendulum = RigidBody("Pendulum")
slider = PrismaticJoint("s", wall, cart, joint_axis=wall.x)
pin = PinJoint("j", cart, pendulum, joint_axis=cart.z,
               child_point=l * pendulum.y)

from sympy import symbols, Function, Derivative, Matrix, sin, cos
t = symbols('t')
l, Pendulum_mass, cart_mass, Pendulum_izz = symbols('l Pendulum_mass cart_mass Pendulum_izz')

q_j = Function('q_j')
u_j = Function('u_j')
u_s = Function('u_s')
M = Matrix([
    [Pendulum_mass*l*u_j(t)**2*sin(q_j(t)) - Pendulum_mass*l*cos(q_j(t))*Derivative(u_j(t), t)
     - (Pendulum_mass + cart_mass)*Derivative(u_s(t), t)],
    [-Pendulum_mass*l*cos(q_j(t))*Derivative(u_s(t), t)
     - (Pendulum_izz + Pendulum_mass*l**2)*Derivative(u_j(t), t)]
])
assert custom_motion(wall,slider, pin) == M