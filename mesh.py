from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from config_loader import PDEConfig, PDEType


@dataclass
class Mesh1D:
    x: np.ndarray
    dx: float
    nx: int


@dataclass
class Mesh2D:
    X: np.ndarray
    Y: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    nx: int
    ny: int


def generate_mesh(config: PDEConfig) -> Mesh1D | Mesh2D:
    if config.dimension == 1:
        return _generate_mesh_1d(config)
    return _generate_mesh_2d(config)


def _generate_mesh_1d(config: PDEConfig) -> Mesh1D:
    d = config.domain_1d
    x = np.linspace(d.x_min, d.x_max, d.nx, dtype=np.float64)
    dx = x[1] - x[0]
    return Mesh1D(x=x, dx=dx, nx=d.nx)


def _generate_mesh_2d(config: PDEConfig) -> Mesh2D:
    d = config.domain_2d
    x = np.linspace(d.x_min, d.x_max, d.nx, dtype=np.float64)
    y = np.linspace(d.y_min, d.y_max, d.ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return Mesh2D(X=X, Y=Y, x=x, y=y, dx=dx, dy=dy, nx=d.nx, ny=d.ny)


def check_stability(config: PDEConfig, mesh: Mesh1D | Mesh2D) -> Tuple[bool, float]:
    if config.pde_type == PDEType.heat:
        return _check_heat_stability(config, mesh)
    elif config.pde_type == PDEType.wave:
        return _check_wave_stability(config, mesh)
    return True, 0.0


def _check_heat_stability(config: PDEConfig, mesh: Mesh1D | Mesh2D) -> Tuple[bool, float]:
    dt = config.time.dt
    alpha = config.alpha
    if config.dimension == 1:
        dx = mesh.dx
        r = alpha * dt / (dx ** 2)
        stable = r <= 0.5
    else:
        dx, dy = mesh.dx, mesh.dy
        r = alpha * dt * (1.0 / dx ** 2 + 1.0 / dy ** 2)
        stable = r <= 0.5
    if not stable:
        warnings.warn(
            f"Heat equation stability condition violated: r={r:.4f} > 0.5. "
            f"Auto-adjusting dt for stability.",
            UserWarning,
        )
    return stable, r


def _check_wave_stability(config: PDEConfig, mesh: Mesh1D | Mesh2D) -> Tuple[bool, float]:
    dt = config.time.dt
    c = config.c
    if config.dimension == 1:
        dx = mesh.dx
        cfl = c * dt / dx
        stable = cfl <= 1.0
    else:
        dx, dy = mesh.dx, mesh.dy
        cfl = c * dt * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2)
        stable = cfl <= 1.0
    if not stable:
        warnings.warn(
            f"Wave equation CFL condition violated: CFL={cfl:.4f} > 1.0. "
            f"Auto-adjusting dt for stability.",
            UserWarning,
        )
    return stable, cfl


def adjust_dt_for_stability(config: PDEConfig, mesh: Mesh1D | Mesh2D) -> float:
    if config.pde_type == PDEType.heat:
        alpha = config.alpha
        if config.dimension == 1:
            dx = mesh.dx
            dt_max = 0.5 * dx ** 2 / alpha
        else:
            dx, dy = mesh.dx, mesh.dy
            dt_max = 0.5 / (alpha * (1.0 / dx ** 2 + 1.0 / dy ** 2))
        return dt_max * 0.95
    elif config.pde_type == PDEType.wave:
        c = config.c
        if config.dimension == 1:
            dx = mesh.dx
            dt_max = dx / c
        else:
            dx, dy = mesh.dx, mesh.dy
            dt_max = 1.0 / (c * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2))
        return dt_max * 0.95
    return config.time.dt


def is_large_scale(config: PDEConfig) -> bool:
    if config.dimension == 1:
        return config.domain_1d.nx >= 1000
    nx = config.domain_2d.nx
    ny = config.domain_2d.ny
    return nx * ny >= 1_000_000
