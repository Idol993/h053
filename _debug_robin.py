import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from mesh import generate_mesh
from solver.poisson import PoissonSolver
import numpy as np

with open("debug_robin_out.txt", "w") as f:
    def P(s=""):
        f.write(s + "\n")

    P("=== Testing _robin_beta in PoissonSolver ===")
    cfg = load_config("test_robin_beta_new.yaml")
    mesh = generate_mesh(cfg)
    solver = PoissonSolver(cfg, mesh)
    bc_left = cfg.boundary_conditions_2d.left
    P(f"  bc_left.type = {bc_left.type}")
    P(f"  bc_left.value = {bc_left.value}")
    P(f"  bc_left.robin_alpha = {bc_left.robin_alpha}")
    P(f"  bc_left.robin_beta = {bc_left.robin_beta}")
    P("")
    P(f"  solver._robin_alpha(bc_left) = {solver._robin_alpha(bc_left)}")
    P(f"  solver._robin_beta(bc_left) = {solver._robin_beta(bc_left)}")
    P("")
    P("=== Testing actual boundary application ===")
    u = np.zeros((40, 40))
    u[1, :] = 0.01
    P(f"  Before: u[0,5] = {u[0, 5]}")
    P(f"  Before: u[1,5] = {u[1, 5]}")
    solver._apply_bc_2d(u)
    P(f"  After:  u[0,5] = {u[0, 5]}")
    P(f"  After:  u[1,5] = {u[1, 5]}")
    alpha = solver._robin_alpha(bc_left)
    beta = solver._robin_beta(bc_left)
    dx = mesh.dx
    P(f"  alpha = {alpha}")
    P(f"  beta = {beta}")
    P(f"  dx = {dx}")
    expected = (u[1, 5] + beta * dx) / (1.0 + alpha * dx)
    P(f"  Expected from formula: (u[1] + beta*dx)/(1+alpha*dx) = {expected:.8f}")
    P(f"  Actual u[0] = {u[0,5]:.8f}")
    P(f"  Match: {np.isclose(u[0,5], expected)}")
    P("")
    P("=== Robin residual check (should be ~0) ===")
    residual = alpha * u[0, :] + (u[0, :] - u[1, :])/dx - beta
    P(f"  Residual max_abs: {np.max(np.abs(residual)):.8f}")
    P(f"  Residual mean:    {np.mean(residual):.8f}")
    P(f"  Residual min:     {np.min(residual):.8f}")
    P(f"  Residual max:     {np.max(residual):.8f}")
    P("")
    P("=== Checking _apply_bc_2d method ===")
    P(f"  bc.left.type = {bc.left.type}")
    P(f"  bc.left.type == BoundaryType.robin: {bc.left.type == 'robin'}")
    P(f"  bc.left.type == BoundaryType.robin: {bc.left.type.value == 'robin'}")
