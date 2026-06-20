from __future__ import annotations

import time
from typing import Optional, Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from config_loader import PDEConfig, BoundaryType
from mesh import Mesh1D, Mesh2D, is_large_scale


class HeatSolver:
    def __init__(self, config: PDEConfig, mesh: Mesh1D | Mesh2D):
        self.config = config
        self.mesh = mesh
        self.method = config.solver.method
        self.alpha = config.alpha
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
            u[:] = 0.0
        elif ic.type == "constant" and ic.value is not None:
            u[:] = ic.value
        elif ic.type == "function" and ic.expression is not None:
            func = _eval_expression(ic.expression)
            u = func(self.mesh.x)
        self._apply_bc_1d(u)
        return u

    def _initial_condition_2d(self) -> np.ndarray:
        u = np.zeros((self.mesh.nx, self.mesh.ny), dtype=np.float64)
        ic = self.config.initial_condition
        if ic is None:
            u[:] = 0.0
        elif ic.type == "constant" and ic.value is not None:
            u[:] = ic.value
        elif ic.type == "function" and ic.expression is not None:
            func = _eval_expression(ic.expression)
            u = func(self.mesh.X, self.mesh.Y)
        self._apply_bc_2d(u)
        return u

    def _bc_value(self, bc_model) -> float:
        if bc_model.type == BoundaryType.dirichlet:
            return bc_model.value
        elif bc_model.type == BoundaryType.neumann:
            return bc_model.value
        return bc_model.value

    def _robin_alpha(self, bc_model) -> float:
        if bc_model.type == BoundaryType.robin and bc_model.robin_alpha is not None:
            return bc_model.robin_alpha
        return 0.0

    def _apply_bc_1d(self, u: np.ndarray):
        bc = self.config.boundary_conditions_1d
        if bc.left.type == BoundaryType.dirichlet:
            u[0] = bc.left.value
        elif bc.left.type == BoundaryType.neumann:
            u[0] = u[1] - bc.left.value * self.mesh.dx
        elif bc.left.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.left)
            beta = bc.left.value
            u[0] = (u[1] + beta * self.mesh.dx) / (1.0 + alpha * self.mesh.dx)

        if bc.right.type == BoundaryType.dirichlet:
            u[-1] = bc.right.value
        elif bc.right.type == BoundaryType.neumann:
            u[-1] = u[-2] + bc.right.value * self.mesh.dx
        elif bc.right.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.right)
            beta = bc.right.value
            u[-1] = (u[-2] + beta * self.mesh.dx) / (1.0 + alpha * self.mesh.dx)

    def _apply_bc_2d(self, u: np.ndarray):
        bc = self.config.boundary_conditions_2d

        if bc.left.type == BoundaryType.dirichlet:
            u[0, :] = bc.left.value
        elif bc.left.type == BoundaryType.neumann:
            u[0, :] = u[1, :] - bc.left.value * self.mesh.dx
        elif bc.left.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.left)
            beta = bc.left.value
            u[0, :] = (u[1, :] + beta * self.mesh.dx) / (1.0 + alpha * self.mesh.dx)

        if bc.right.type == BoundaryType.dirichlet:
            u[-1, :] = bc.right.value
        elif bc.right.type == BoundaryType.neumann:
            u[-1, :] = u[-2, :] + bc.right.value * self.mesh.dx
        elif bc.right.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.right)
            beta = bc.right.value
            u[-1, :] = (u[-2, :] + beta * self.mesh.dx) / (1.0 + alpha * self.mesh.dx)

        if bc.bottom.type == BoundaryType.dirichlet:
            u[:, 0] = bc.bottom.value
        elif bc.bottom.type == BoundaryType.neumann:
            u[:, 0] = u[:, 1] - bc.bottom.value * self.mesh.dy
        elif bc.bottom.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.bottom)
            beta = bc.bottom.value
            u[:, 0] = (u[:, 1] + beta * self.mesh.dy) / (1.0 + alpha * self.mesh.dy)

        if bc.top.type == BoundaryType.dirichlet:
            u[:, -1] = bc.top.value
        elif bc.top.type == BoundaryType.neumann:
            u[:, -1] = u[:, -2] + bc.top.value * self.mesh.dy
        elif bc.top.type == BoundaryType.robin:
            alpha = self._robin_alpha(bc.top)
            beta = bc.top.value
            u[:, -1] = (u[:, -2] + beta * self.mesh.dy) / (1.0 + alpha * self.mesh.dy)

    def _solve_1d(self, result_buffer=None) -> list[np.ndarray]:
        u = self._initial_condition_1d()
        if result_buffer is not None:
            result_buffer.append(u)
            self.results = result_buffer
        else:
            self.results = [u.copy()]
        method = self.method
        if method == "auto":
            r = self.alpha * self.dt / (self.mesh.dx ** 2)
            method = "explicit" if r <= 0.5 else "implicit"

        start = time.time()
        with tqdm(total=self.n_steps, desc="Heat 1D", unit="step") as pbar:
            for step in range(self.n_steps):
                if method == "explicit":
                    u = self._step_1d_explicit(u)
                else:
                    u = self._step_1d_implicit(u)
                self._apply_bc_1d(u)
                if result_buffer is not None:
                    result_buffer.append(u)
                    if hasattr(result_buffer, 'flush'):
                        result_buffer.flush()
                else:
                    self.results.append(u.copy())
                elapsed = time.time() - start
                remaining = elapsed / (step + 1) * (self.n_steps - step - 1)
                pbar.set_postfix(elapsed=f"{elapsed:.1f}s", eta=f"{remaining:.1f}s")
                pbar.update(1)
        if result_buffer is not None:
            return result_buffer
        return self.results

    def _step_1d_explicit(self, u: np.ndarray) -> np.ndarray:
        r = self.alpha * self.dt / (self.mesh.dx ** 2)
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        return u_new

    def _step_1d_implicit(self, u: np.ndarray) -> np.ndarray:
        n = self.mesh.nx - 2
        r = self.alpha * self.dt / (self.mesh.dx ** 2)
        diag = np.full(n, 1 + 2 * r)
        off_diag = np.full(n - 1, -r)
        rhs = u[1:-1].copy()
        bc = self.config.boundary_conditions_1d
        rhs[0] += r * self._bc_value(bc.left)
        rhs[-1] += r * self._bc_value(bc.right)
        u_new = u.copy()
        u_new[1:-1] = thomas_solve(diag, off_diag, off_diag.copy(), rhs)
        return u_new

    def _solve_2d(self, result_buffer=None) -> list[np.ndarray]:
        u = self._initial_condition_2d()
        if result_buffer is not None:
            result_buffer.append(u)
            self.results = result_buffer
        else:
            self.results = [u.copy()]
        method = self.method
        if method == "auto":
            rx = self.alpha * self.dt / (self.mesh.dx ** 2)
            ry = self.alpha * self.dt / (self.mesh.dy ** 2)
            method = "explicit" if (rx + ry) <= 0.5 else "implicit"

        large = is_large_scale(self.config)
        start = time.time()
        with tqdm(total=self.n_steps, desc="Heat 2D", unit="step") as pbar:
            for step in range(self.n_steps):
                if method == "explicit":
                    u = self._step_2d_explicit(u)
                else:
                    u = self._step_2d_implicit(u, use_sparse=large)
                self._apply_bc_2d(u)
                if result_buffer is not None:
                    result_buffer.append(u)
                    if hasattr(result_buffer, 'flush'):
                        result_buffer.flush()
                else:
                    self.results.append(u.copy())
                elapsed = time.time() - start
                remaining = elapsed / (step + 1) * (self.n_steps - step - 1)
                pbar.set_postfix(elapsed=f"{elapsed:.1f}s", eta=f"{remaining:.1f}s")
                pbar.update(1)
        if result_buffer is not None:
            return result_buffer
        return self.results

    def _step_2d_explicit(self, u: np.ndarray) -> np.ndarray:
        rx = self.alpha * self.dt / (self.mesh.dx ** 2)
        ry = self.alpha * self.dt / (self.mesh.dy ** 2)
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + rx * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
            + ry * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
        )
        return u_new

    def _step_2d_implicit(self, u: np.ndarray, use_sparse: bool = False) -> np.ndarray:
        nx, ny = self.mesh.nx, self.mesh.ny
        rx = self.alpha * self.dt / (self.mesh.dx ** 2)
        ry = self.alpha * self.dt / (self.mesh.dy ** 2)
        n = (nx - 2) * (ny - 2)

        bc = self.config.boundary_conditions_2d

        if use_sparse:
            return self._step_2d_implicit_sparse(u, rx, ry, nx, ny, bc)

        if n <= 0:
            return u.copy()

        A = _build_2d_heat_matrix_dense(rx, ry, nx - 2, ny - 2)
        rhs = u[1:-1, 1:-1].flatten().copy()
        rhs = _apply_2d_bc_rhs(rhs, rx, ry, nx - 2, ny - 2, bc)
        sol = np.linalg.solve(A, rhs)
        u_new = u.copy()
        u_new[1:-1, 1:-1] = sol.reshape(nx - 2, ny - 2)
        return u_new

    def _step_2d_implicit_sparse(self, u, rx, ry, nx, ny, bc):
        n = (nx - 2) * (ny - 2)
        if n <= 0:
            return u.copy()
        A = _build_2d_heat_matrix_sparse(rx, ry, nx - 2, ny - 2)
        rhs = u[1:-1, 1:-1].flatten().copy()
        rhs = _apply_2d_bc_rhs(rhs, rx, ry, nx - 2, ny - 2, bc)
        sol = spsolve(A, rhs)
        u_new = u.copy()
        u_new[1:-1, 1:-1] = sol.reshape(nx - 2, ny - 2)
        return u_new


def thomas_solve(
    diag: np.ndarray, upper: np.ndarray, lower: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    n = len(diag)
    c = np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)

    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]

    for i in range(1, n):
        m = diag[i] - lower[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = upper[i] / m
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / m

    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


def _build_2d_heat_matrix_dense(rx, ry, ni, nj):
    n = ni * nj
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(ni):
        for j in range(nj):
            k = i * nj + j
            A[k, k] = 1 + 2 * rx + 2 * ry
            if i > 0:
                A[k, k - nj] = -rx
            if i < ni - 1:
                A[k, k + nj] = -rx
            if j > 0:
                A[k, k - 1] = -ry
            if j < nj - 1:
                A[k, k + 1] = -ry
    return A


def _build_2d_heat_matrix_sparse(rx, ry, ni, nj):
    n = ni * nj
    rows, cols, vals = [], [], []
    for i in range(ni):
        for j in range(nj):
            k = i * nj + j
            rows.append(k)
            cols.append(k)
            vals.append(1 + 2 * rx + 2 * ry)
            if i > 0:
                rows.append(k)
                cols.append(k - nj)
                vals.append(-rx)
            if i < ni - 1:
                rows.append(k)
                cols.append(k + nj)
                vals.append(-rx)
            if j > 0:
                rows.append(k)
                cols.append(k - 1)
                vals.append(-ry)
            if j < nj - 1:
                rows.append(k)
                cols.append(k + 1)
                vals.append(-ry)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _apply_2d_bc_rhs(rhs, rx, ry, ni, nj, bc):
    for j in range(nj):
        k = 0 * nj + j
        rhs[k] += rx * _bc_val(bc.left)
        k = (ni - 1) * nj + j
        rhs[k] += rx * _bc_val(bc.right)
    for i in range(ni):
        k = i * nj + 0
        rhs[k] += ry * _bc_val(bc.bottom)
        k = i * nj + (nj - 1)
        rhs[k] += ry * _bc_val(bc.top)
    return rhs


def _bc_val(bc_model) -> float:
    if bc_model.type == BoundaryType.dirichlet:
        return bc_model.value
    return 0.0


def _eval_expression(expr: str) -> Callable:
    import numpy as _np

    allowed = {"np": _np, "sin": _np.sin, "cos": _np.cos, "exp": _np.exp,
               "pi": _np.pi, "sqrt": _np.sqrt, "abs": _np.abs,
               "x": None, "y": None, "t": None}
    code = compile(expr, "<initial_condition>", "eval")

    def func(*args):
        local = dict(allowed)
        if len(args) == 1:
            local["x"] = args[0]
        elif len(args) == 2:
            local["x"] = args[0]
            local["y"] = args[1]
        return eval(code, {"__builtins__": {}}, local)

    return func
