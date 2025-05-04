# library: sympy
# version: 1.11
# extra_dependencies: []
import sympy.physics.quantum
import sympy


def custom_trace(n: int) -> sympy.physics.quantum.trace.Tr:
    return sympy.physics.quantum.trace.Tr(n)
