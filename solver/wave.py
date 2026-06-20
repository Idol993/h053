from __future__ import annotations

import time
from typing import Callable

import numpy as np
from tqdm import tqdm

from config_loader import PDEConfig, BoundaryType
from mesh import Mesh1D, Mesh2D
from solver.heat import _eval_expression


class WaveSolver:
    def __init__(self, config: PDEConfig, mesh: Mesh1D | Mesh2D):
        self.config = config
        self.mesh = mesh
        self.c = config.c
        self.dt = config.time.dt
        self.n_steps = config.time.n_steps
        self.results: list[np.ndarray] = []

    def solve(self, result_buffer=None) -> list[np.ndarray]:
        if self.config.dimension == 1:
            return self._solve_1d(result_buffer)
        return self._solve_2d(result_buffer)

    def _initial_condition_1d(self) -> np.ndarray:
        u = np.zeros(self.mesh.nx, dtype=np.float64)
        ic = self.config.initial_condition
        if ic is None:
            return u
        if ic.type == "constant" and ic.value is not None:
            u[:] = ic.value
        elif ic.type == "function" and ic.expression is not None:
            func = _eval_expression(ic.expression)
            u = func(self.mesh.x)
        return u

    def _initial_velocity_1d(self) -> np.ndarray:
        v = np.zeros(self.mesh.nx, dtype=np.float64)
        ic = self.config.initial_velocity
        if ic is None:
            return v
        if ic.type == "constant" and ic.value is not None:
            v[:] = ic.value
        elif ic.type == "function" and ic.expression is not None:
            func = _eval_expression(ic.expression)
            v = func(self.mesh.x)
        return v

    def _initial_condition_2d(self) -> np.ndarray:
        u = np.zeros((self.mesh.nx, self.mesh.ny), dtype=np.float64)
        ic = self.config.initial_condition
        if ic is None:
            return u
        if ic.type == "constant" and ic.value is not None:
            u[:] = ic.value
        elif ic.type == "function" and ic.expression is not None:
            func = _eval_expression(ic.expression)
            u = func(self.mesh.X, self.mesh.Y)
        return u

    def _initial_velocity_2d(self) -> np.ndarray:
        v = np.zeros((self.mesh.nx, self.mesh.ny), dtype=np.float64)
        ic = self.config.initial_velocity
        if ic is None:
            return v
        if ic.type == "constant" and ic.value is not None:
            v[:] = ic.value
        elif ic.type == "function" and ic.expression is not None:
            func = _eval_expression(ic.expression)
            v = func(self.mesh.X, self.mesh.Y)
        return v

    def _bc_value(self, bc_model) -> float:
        if bc_model.type == BoundaryType.dirichlet:
            return bc_model.value
        return 0.0

    def _robin_alpha(self, bc_model) -> float:
        if bc_model.type == BoundaryType.robin and bc_model.robin_alpha is not None:
            return bc_model.robin_alpha
        return 0.0

    def _robin_beta(self, bc_model) -> float:
        if bc_model.type == BoundaryType.robin:
            if bc_model.robin_beta is not None:
                return bc_model.robin_beta
            return bc_model.value
        return 0.0

    def _apply_bc_1d(self, u: np.ndarray):
        bc = self.config.boundary_conditions_1d
        if bc.left.type == BoundaryType.dirichlet:
            u[0] = bc.left.value
        elif bc.left.type == BoundaryType.neumann:
            u[0] = u[1] - bc.left.value * self.mesh.dx
        elif bc.left.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.left)
            beta = self._robin_beta(bc.left)
            u[0] = (u[1] + beta * self.mesh.dx) / (1.0 + alpha * self.mesh.dx)

        if bc.right.type == BoundaryType.dirichlet:
            u[-1] = bc.right.value
        elif bc.right.type == BoundaryType.neumann:
            u[-1] = u[-2] + bc.right.value * self.mesh.dx
        elif bc.right.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.right)
            beta = self._robin_beta(bc.right)
            u[-1] = (u[-2] + beta * self.mesh.dx) / (1.0 + alpha * self.mesh.dx)

    def _apply_bc_2d(self, u: np.ndarray):
        bc = self.config.boundary_conditions_2d

        if bc.left.type == BoundaryType.dirichlet:
            u[0, :] = bc.left.value
        elif bc.left.type == BoundaryType.neumann:
            u[0, :] = u[1, :] - bc.left.value * self.mesh.dx
        elif bc.left.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.left)
            beta = self._robin_beta(bc.left)
            u[0, :] = (u[1, :] + beta * self.mesh.dx) / (1.0 + alpha * self.mesh.dx)

        if bc.right.type == BoundaryType.dirichlet:
            u[-1, :] = bc.right.value
        elif bc.right.type == BoundaryType.neumann:
            u[-1, :] = u[-2, :] + bc.right.value * self.mesh.dx
        elif bc.right.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.right)
            beta = self._robin_beta(bc.right)
            u[-1, :] = (u[-2, :] + beta * self.mesh.dx) / (1.0 + alpha * self.mesh.dx)

        if bc.bottom.type == BoundaryType.dirichlet:
            u[:, 0] = bc.bottom.value
        elif bc.bottom.type == BoundaryType.neumann:
            u[:, 0] = u[:, 1] - bc.bottom.value * self.mesh.dy
        elif bc.bottom.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.bottom)
            beta = self._robin_beta(bc.bottom)
            u[:, 0] = (u[:, 1] + beta * self.mesh.dy) / (1.0 + alpha * self.mesh.dy)

        if bc.top.type == BoundaryType.dirichlet:
            u[:, -1] = bc.top.value
        elif bc.top.type == BoundaryType.neumann:
            u[:, -1] = u[:, -2] + bc.top.value * self.mesh.dy
        elif bc.top.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.top)
            beta = self._robin_beta(bc.top)
            u[:, -1] = (u[:, -2] + beta * self.mesh.dy) / (1.0 + alpha * self.mesh.dy)

    def _solve_1d(self, result_buffer=None) -> list[np.ndarray]:
        u_prev = self._initial_condition_1d()
        v0 = self._initial_velocity_1d()
        self._apply_bc_1d(u_prev)

        r2 = (self.c * self.dt / self.mesh.dx) ** 2
        u_curr = np.copy(u_prev)
        u_curr[1:-1] = (
            u_prev[1:-1]
            + self.dt * v0[1:-1]
            + 0.5 * r2 * (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2])
        )
        self._apply_bc_1d(u_curr)

        if result_buffer is not None:
            result_buffer.append(u_prev)
            result_buffer.append(u_curr)
            if hasattr(result_buffer, 'flush'):
                result_buffer.flush()
            self.results = result_buffer
        else:
            self.results = [u_prev.copy(), u_curr.copy()]

        start = time.time()
        with tqdm(total=self.n_steps - 1, desc="Wave 1D", unit="step") as pbar:
            for step in range(1, self.n_steps):
                u_next = np.copy(u_curr)
                u_next[1:-1] = (
                    2 * u_curr[1:-1]
                    - u_prev[1:-1]
                    + r2 * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2])
                )
                self._apply_bc_1d(u_next)
                if result_buffer is not None:
                    result_buffer.append(u_next)
                    if hasattr(result_buffer, 'flush'):
                        result_buffer.flush()
                else:
                    self.results.append(u_next.copy())
                u_prev = u_curr
                u_curr = u_next
                elapsed = time.time() - start
                remaining = elapsed / step * (self.n_steps - 1 - step)
                pbar.set_postfix(elapsed=f"{elapsed:.1f}s", eta=f"{remaining:.1f}s")
                pbar.update(1)
        if result_buffer is not None:
            return result_buffer
        return self.results

    def _solve_2d(self, result_buffer=None) -> list[np.ndarray]:
        u_prev = self._initial_condition_2d()
        v0 = self._initial_velocity_2d()
        self._apply_bc_2d(u_prev)

        rx2 = (self.c * self.dt / self.mesh.dx) ** 2
        ry2 = (self.c * self.dt / self.mesh.dy) ** 2

        u_curr = np.copy(u_prev)
        u_curr[1:-1, 1:-1] = (
            u_prev[1:-1, 1:-1]
            + self.dt * v0[1:-1, 1:-1]
            + 0.5 * rx2 * (u_prev[2:, 1:-1] - 2 * u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1])
            + 0.5 * ry2 * (u_prev[1:-1, 2:] - 2 * u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2])
        )
        self._apply_bc_2d(u_curr)

        if result_buffer is not None:
            result_buffer.append(u_prev)
            result_buffer.append(u_curr)
            if hasattr(result_buffer, 'flush'):
                result_buffer.flush()
            self.results = result_buffer
        else:
            self.results = [u_prev.copy(), u_curr.copy()]

        start = time.time()
        with tqdm(total=self.n_steps - 1, desc="Wave 2D", unit="step") as pbar:
            for step in range(1, self.n_steps):
                u_next = np.copy(u_curr)
                u_next[1:-1, 1:-1] = (
                    2 * u_curr[1:-1, 1:-1]
                    - u_prev[1:-1, 1:-1]
                    + rx2 * (u_curr[2:, 1:-1] - 2 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1])
                    + ry2 * (u_curr[1:-1, 2:] - 2 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2])
                )
                self._apply_bc_2d(u_next)
                if result_buffer is not None:
                    result_buffer.append(u_next)
                    if hasattr(result_buffer, 'flush'):
                        result_buffer.flush()
                else:
                    self.results.append(u_next.copy())
                u_prev = u_curr
                u_curr = u_next
                elapsed = time.time() - start
                remaining = elapsed / step * (self.n_steps - 1 - step)
                pbar.set_postfix(elapsed=f"{elapsed:.1f}s", eta=f"{remaining:.1f}s")
                pbar.update(1)
        if result_buffer is not None:
            return result_buffer
        return self.results
