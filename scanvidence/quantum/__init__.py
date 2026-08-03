"""QUBO feature-selection formulation and solvers (IBM Quantum Open Plan,
D-Wave Ocean SDK, PennyLane) with classical simulated annealing / genetic
algorithm as the always-available fallback. Stretch objective — nothing
in the core pipeline imports from this package.
"""

from .base import BaseSelector
