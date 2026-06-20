from solver.heat import HeatSolver
from solver.wave import WaveSolver
from solver.poisson import PoissonSolver

SOLVER_REGISTRY = {
    "heat": HeatSolver,
    "wave": WaveSolver,
    "poisson": PoissonSolver,
}

__all__ = ["HeatSolver", "WaveSolver", "PoissonSolver", "SOLVER_REGISTRY"]
