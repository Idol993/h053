from __future__ import annotations

import time
from typing import Callable

import numpy as np
from tqdm import tqdm

from config_loader import PDEConfig, BoundaryType
from mesh import Mesh1D, Mesh2D
from solver.heat import _eval_expression


class PoissonSolver:
    def __init__(self, config: PDEConfig, mesh: Mesh1D | Mesh2D):
        self.config = config
        self.mesh = mesh
        self.max_iterations = config.solver.max_iterations
        self.threshold = config.solver.convergence_threshold
        self.source_expr = config.source
        self.results: list[np.ndarray] = []

    def solve(self) -> list[np.ndarray]:
        if self.config.dimension == 1:
            return self._solve_1d()
        return self._solve_2d()

    def _source_1d(self, x: np.ndarray) -> np.ndarray:
        if self.source_expr is None:
            return np.zeros_like(x)
        func = _eval_expression(self.source_expr)
        return func(x)

    def _source_2d(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.source_expr is None:
            return np.zeros_like(X)
        func = _eval_expression(self.source_expr)
        return func(X, Y)

    def _bc_value(self, bc_model) -> float:
        if bc_model.type == BoundaryType.dirichlet:
            return bc_model.value
        return 0.0

    def _apply_bc_1d(self, u: np.ndarray):
        bc = self.config.boundary_conditions_1d
        u[0] = self._bc_value(bc.left)
        u[-1] = self._bc_value(bc.right)
        if bc.left.type == BoundaryType.neumann:
            u[0] = u[1] - bc.left.value * self.mesh.dx
        if bc.right.type == BoundaryType.neumann:
            u[-1] = u[-2] + bc.right.value * self.mesh.dx

    def _apply_bc_2d(self, u: np.ndarray):
        bc = self.config.boundary_conditions_2d
        u[0, :] = self._bc_value(bc.left)
        u[-1, :] = self._bc_value(bc.right)
        u[:, 0] = self._bc_value(bc.bottom)
        u[:, -1] = self._bc_value(bc.top)
        if bc.left.type == BoundaryType.neumann:
            u[0, :] = u[1, :] - bc.left.value * self.mesh.dx
        if bc.right.type == BoundaryType.neumann:
            u[-1, :] = u[-2, :] + bc.right.value * self.mesh.dx
        if bc.bottom.type == BoundaryType.neumann:
            u[:, 0] = u[:, 1] - bc.bottom.value * self.mesh.dy
        if bc.top.type == BoundaryType.neumann:
            u[:, -1] = u[:, -2] + bc.top.value * self.mesh.dy

    def _solve_1d(self) -> list[np.ndarray]:
        nx = self.mesh.nx
        dx = self.mesh.dx
        u = np.zeros(nx, dtype=np.float64)
        self._apply_bc_1d(u)
        f = self._source_1d(self.mesh.x)

        start = time.time()
        with tqdm(total=self.max_iterations, desc="Poisson 1D Gauss-Seidel", unit="iter") as pbar:
            for it in range(self.max_iterations):
                u_old = u.copy()
                for i in range(1, nx - 1):
                    u[i] = 0.5 * (u[i - 1] + u[i + 1] - dx ** 2 * f[i])
                self._apply_bc_1d(u)
                residual = np.max(np.abs(u - u_old))
                if residual < self.threshold:
                    pbar.set_postfix(residual=f"{residual:.2e}", converged="Yes",
                                     elapsed=f"{time.time() - start:.1f}s")
                    pbar.update(self.max_iterations - it)
                    break
                elapsed = time.time() - start
                remaining = elapsed / (it + 1) * (self.max_iterations - it - 1)
                pbar.set_postfix(residual=f"{residual:.2e}", elapsed=f"{elapsed:.1f}s",
                                 eta=f"{remaining:.1f}s")
                pbar.update(1)

        self.results = [u]
        return self.results

    def _solve_2d(self) -> list[np.ndarray]:
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        u = np.zeros((nx, ny), dtype=np.float64)
        self._apply_bc_2d(u)
        f = self._source_2d(self.mesh.X, self.mesh.Y)

        coeff = 2.0 * (1.0 / dx ** 2 + 1.0 / dy ** 2)

        start = time.time()
        with tqdm(total=self.max_iterations, desc="Poisson 2D Gauss-Seidel", unit="iter") as pbar:
            for it in range(self.max_iterations):
                u_old = u.copy()
                for i in range(1, nx - 1):
                    for j in range(1, ny - 1):
                        u[i, j] = (
                            (u[i - 1, j] + u[i + 1, j]) / dx ** 2
                            + (u[i, j - 1] + u[i, j + 1]) / dy ** 2
                            - f[i, j]
                        ) / coeff
                self._apply_bc_2d(u)
                residual = np.max(np.abs(u - u_old))
                if residual < self.threshold:
                    pbar.set_postfix(residual=f"{residual:.2e}", converged="Yes",
                                     elapsed=f"{time.time() - start:.1f}s")
                    pbar.update(self.max_iterations - it)
                    break
                elapsed = time.time() - start
                remaining = elapsed / (it + 1) * (self.max_iterations - it - 1)
                pbar.set_postfix(residual=f"{residual:.2e}", elapsed=f"{elapsed:.1f}s",
                                 eta=f"{remaining:.1f}s")
                pbar.update(1)

        self.results = [u]
        return self.results
