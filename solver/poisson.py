from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from config_loader import PDEConfig, BoundaryType
from mesh import Mesh1D, Mesh2D, is_large_scale
from solver.heat import _eval_expression


class PoissonSolver:
    def __init__(self, config: PDEConfig, mesh: Mesh1D | Mesh2D):
        self.config = config
        self.mesh = mesh
        self.max_iterations = config.solver.max_iterations
        self.threshold = config.solver.convergence_threshold
        self.source_expr = config.source
        self.results: list[np.ndarray] = []
        self._use_sparse = is_large_scale(config)

    def solve(self, result_buffer=None) -> list[np.ndarray]:
        if self.config.dimension == 1:
            return self._solve_1d(result_buffer)
        return self._solve_2d(result_buffer)

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
        nx = self.mesh.nx
        dx = self.mesh.dx
        bc = self.config.boundary_conditions_1d
        f = self._source_1d(self.mesh.x)

        has_dirichlet = (
            bc.left.type == BoundaryType.dirichlet
            or bc.right.type == BoundaryType.dirichlet
        )

        if self._use_sparse or not has_dirichlet:
            tqdm.write("[Poisson 1D] Using direct sparse solver (large scale / non-Dirichlet BC)")
            u = self._solve_1d_direct(f)
            if result_buffer is not None:
                result_buffer.append(u)
                self.results = result_buffer
            else:
                self.results = [u]
            return self.results

        u = np.zeros(nx, dtype=np.float64)
        self._apply_bc_1d(u)

        start = time.time()
        with tqdm(total=self.max_iterations, desc="Poisson 1D Gauss-Seidel", unit="iter") as pbar:
            for it in range(self.max_iterations):
                u_old = u.copy()
                for i in range(1, nx - 1):
                    u[i] = 0.5 * (u[i - 1] + u[i + 1] + dx ** 2 * f[i])
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

        if result_buffer is not None:
            result_buffer.append(u)
            self.results = result_buffer
        else:
            self.results = [u]
        return self.results

    def _solve_1d_direct(self, f: np.ndarray) -> np.ndarray:
        nx = self.mesh.nx
        dx = self.mesh.dx
        bc = self.config.boundary_conditions_1d
        u = np.zeros(nx, dtype=np.float64)

        if (
            bc.left.type == BoundaryType.dirichlet
            and bc.right.type == BoundaryType.dirichlet
        ):
            n = nx - 2
            diag = np.full(n, -2.0)
            off = np.full(n - 1, 1.0)
            A = sparse.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr")
            rhs = -dx ** 2 * f[1:-1].copy()
            rhs[0] -= bc.left.value
            rhs[-1] -= bc.right.value
            u[1:-1] = spsolve(A, rhs)
            u[0] = bc.left.value
            u[-1] = bc.right.value
        else:
            n = nx
            diag = np.full(n, -2.0)
            off = np.ones(n - 1)
            A = sparse.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="lil")
            rhs = -dx ** 2 * f.copy()

            if bc.left.type == BoundaryType.dirichlet:
                A[0, :] = 0
                A[0, 0] = 1.0
                rhs[0] = bc.left.value
            elif bc.left.type == BoundaryType.neumann:
                A[0, 0] = -1.0
                A[0, 1] = 1.0
                rhs[0] = bc.left.value * dx
            elif bc.left.type == BoundaryType.robin:
                alpha = self._robin_alpha(bc.left)
                beta = bc.left.value
                A[0, 0] = 1.0 + alpha * dx
                A[0, 1] = -1.0
                rhs[0] = beta * dx

            if bc.right.type == BoundaryType.dirichlet:
                A[-1, :] = 0
                A[-1, -1] = 1.0
                rhs[-1] = bc.right.value
            elif bc.right.type == BoundaryType.neumann:
                A[-1, -2] = -1.0
                A[-1, -1] = 1.0
                rhs[-1] = bc.right.value * dx
            elif bc.right.type == BoundaryType.robin:
                alpha = self._robin_alpha(bc.right)
                beta = bc.right.value
                A[-1, -2] = -1.0
                A[-1, -1] = 1.0 + alpha * dx
                rhs[-1] = beta * dx

            A = A.tocsr()
            u[:] = spsolve(A, rhs)

        return u

    def _solve_2d(self, result_buffer=None) -> list[np.ndarray]:
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        bc = self.config.boundary_conditions_2d
        f = self._source_2d(self.mesh.X, self.mesh.Y)

        if self._use_sparse:
            tqdm.write(f"[Poisson 2D] Large scale detected ({nx}x{ny}), using sparse direct solver")
            u = self._solve_2d_sparse(f)
            if result_buffer is not None:
                result_buffer.append(u)
                self.results = result_buffer
            else:
                self.results = [u]
            return self.results

        u = np.zeros((nx, ny), dtype=np.float64)
        self._apply_bc_2d(u)

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
                            + f[i, j]
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

        if result_buffer is not None:
            result_buffer.append(u)
            self.results = result_buffer
        else:
            self.results = [u]
        return self.results

    def _solve_2d_sparse(self, f: np.ndarray) -> np.ndarray:
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        bc = self.config.boundary_conditions_2d
        n = nx * ny

        rows, cols, vals = [], [], []
        rhs = np.zeros(n, dtype=np.float64)

        dx2 = dx ** 2
        dy2 = dy ** 2
        diag_val = -(2.0 / dx2 + 2.0 / dy2)
        x_val = 1.0 / dx2
        y_val = 1.0 / dy2

        def idx(i, j):
            return i * ny + j

        for i in range(nx):
            for j in range(ny):
                k = idx(i, j)

                if i == 0:
                    if bc.left.type == BoundaryType.dirichlet:
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0)
                        rhs[k] = bc.left.value
                    elif bc.left.type == BoundaryType.neumann:
                        rows.append(k)
                        cols.append(k)
                        vals.append(-1.0)
                        rows.append(k)
                        cols.append(idx(1, j))
                        vals.append(1.0)
                        rhs[k] = bc.left.value * dx
                    elif bc.left.type == BoundaryType.robin:
                        alpha = self._robin_alpha(bc.left)
                        beta = bc.left.value
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0 + alpha * dx)
                        rows.append(k)
                        cols.append(idx(1, j))
                        vals.append(-1.0)
                        rhs[k] = beta * dx
                elif i == nx - 1:
                    if bc.right.type == BoundaryType.dirichlet:
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0)
                        rhs[k] = bc.right.value
                    elif bc.right.type == BoundaryType.neumann:
                        rows.append(k)
                        cols.append(idx(nx - 2, j))
                        vals.append(-1.0)
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0)
                        rhs[k] = bc.right.value * dx
                    elif bc.right.type == BoundaryType.robin:
                        alpha = self._robin_alpha(bc.right)
                        beta = bc.right.value
                        rows.append(k)
                        cols.append(idx(nx - 2, j))
                        vals.append(-1.0)
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0 + alpha * dx)
                        rhs[k] = beta * dx
                elif j == 0:
                    if bc.bottom.type == BoundaryType.dirichlet:
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0)
                        rhs[k] = bc.bottom.value
                    elif bc.bottom.type == BoundaryType.neumann:
                        rows.append(k)
                        cols.append(k)
                        vals.append(-1.0)
                        rows.append(k)
                        cols.append(idx(i, 1))
                        vals.append(1.0)
                        rhs[k] = bc.bottom.value * dy
                    elif bc.bottom.type == BoundaryType.robin:
                        alpha = self._robin_alpha(bc.bottom)
                        beta = bc.bottom.value
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0 + alpha * dy)
                        rows.append(k)
                        cols.append(idx(i, 1))
                        vals.append(-1.0)
                        rhs[k] = beta * dy
                elif j == ny - 1:
                    if bc.top.type == BoundaryType.dirichlet:
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0)
                        rhs[k] = bc.top.value
                    elif bc.top.type == BoundaryType.neumann:
                        rows.append(k)
                        cols.append(idx(i, ny - 2))
                        vals.append(-1.0)
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0)
                        rhs[k] = bc.top.value * dy
                    elif bc.top.type == BoundaryType.robin:
                        alpha = self._robin_alpha(bc.top)
                        beta = bc.top.value
                        rows.append(k)
                        cols.append(idx(i, ny - 2))
                        vals.append(-1.0)
                        rows.append(k)
                        cols.append(k)
                        vals.append(1.0 + alpha * dy)
                        rhs[k] = beta * dy
                else:
                    rows.append(k)
                    cols.append(k)
                    vals.append(diag_val)
                    rows.append(k)
                    cols.append(idx(i - 1, j))
                    vals.append(x_val)
                    rows.append(k)
                    cols.append(idx(i + 1, j))
                    vals.append(x_val)
                    rows.append(k)
                    cols.append(idx(i, j - 1))
                    vals.append(y_val)
                    rows.append(k)
                    cols.append(idx(i, j + 1))
                    vals.append(y_val)
                    rhs[k] = -f[i, j]

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        sol = spsolve(A, rhs)
        u = sol.reshape(nx, ny)
        return u
