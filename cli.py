from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import click
import numpy as np

from config_loader import PDEConfig, PDEType, load_config
from mesh import Mesh1D, Mesh2D, adjust_dt_for_stability, check_stability, generate_mesh
from solver import SOLVER_REGISTRY
from exporter import ResultExporter, ResultBuffer
from visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="pde-solver")
def cli():
    """PDE Numerical Solver CLI - Solve heat, wave, and Poisson equations using finite difference methods."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./results",
    help="Output directory for results.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.STRING,
    default="csv+mp4",
    help="Output format: csv, mp4, csv+mp4, csv+png, png.",
)
@click.option(
    "--step-interval",
    type=int,
    default=10,
    help="Interval between saved steps for PNG/CSV output.",
)
@click.option(
    "--fps",
    type=int,
    default=10,
    help="Frames per second for MP4 animation.",
)
@click.option(
    "--use-nvenc",
    is_flag=True,
    default=False,
    help="Use NVIDIA hardware encoding for MP4.",
)
@click.option(
    "--use-memmap",
    is_flag=True,
    default=False,
    help="Use numpy.memmap for large results (reduces memory usage).",
)
@click.option(
    "--sample",
    is_flag=True,
    default=False,
    help="Export sample slices (centerlines, boundary layers, stats) instead of full CSV for 2D large grids.",
)
@click.option(
    "--summary/--no-summary",
    default=True,
    help="Export summary.json with problem stats and boundary residuals.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def solve(
    config: str,
    output: str,
    output_format: str,
    step_interval: int,
    fps: int,
    use_nvenc: bool,
    use_memmap: bool,
    sample: bool,
    summary: bool,
    verbose: bool,
):
    """Solve a PDE based on the provided YAML configuration."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo("=" * 60)
    click.echo("  PDE Numerical Solver")
    click.echo("=" * 60)

    try:
        pde_config = load_config(config)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    click.echo(f"\nPDE Type     : {pde_config.pde_type.value}")
    click.echo(f"Dimension    : {pde_config.dimension}D")

    mesh = generate_mesh(pde_config)
    if pde_config.dimension == 1:
        click.echo(f"Grid         : {mesh.nx} points, dx={mesh.dx:.6f}")
    else:
        click.echo(f"Grid         : {mesh.nx}x{mesh.ny}, dx={mesh.dx:.6f}, dy={mesh.dy:.6f}")

    if pde_config.time:
        click.echo(f"Time steps   : {pde_config.time.n_steps}, dt={pde_config.time.dt:.6f}")

    stable, param = check_stability(pde_config, mesh)
    if not stable:
        new_dt = adjust_dt_for_stability(pde_config, mesh)
        click.echo(f"\nWARNING: Stability condition violated (parameter={param:.4f})")
        click.echo(f"  Auto-adjusting dt: {pde_config.time.dt:.6f} -> {new_dt:.6f}")
        pde_config.time.dt = new_dt
        pde_config.time.n_steps = int(pde_config.time.t_max / new_dt)
        click.echo(f"  New total steps: {pde_config.time.n_steps}")
    else:
        click.echo(f"Stability    : OK (parameter={param:.4f})")

    solver_cls = SOLVER_REGISTRY[pde_config.pde_type.value]
    solver = solver_cls(pde_config, mesh)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    result_buffer = None
    if use_memmap:
        result_buffer = ResultBuffer.from_config(
            pde_config, output_path, use_memmap=True
        )
        click.echo(f"\nMemmap mode: results streaming to {result_buffer.memmap_path}")
        memmap_size_mb = (
            np.prod(result_buffer.full_shape) * 4 / (1024 * 1024)
        )
        click.echo(f"  Expected size: {memmap_size_mb:.1f} MB (float32)")

    click.echo(f"\nSolver       : {solver_cls.__name__}")
    click.echo(f"Output dir   : {output}")
    click.echo(f"Output format: {output_format}")
    click.echo("-" * 60)
    click.echo("Starting solve...")

    solve_start = time.time()
    results = solver.solve(result_buffer=result_buffer)
    solve_time = time.time() - solve_start

    if use_memmap and result_buffer is not None:
        result_buffer.flush()
        size_mb = result_buffer.memmap_path.stat().st_size / (1024 * 1024)
        click.echo(f"\nMemmap results saved: {result_buffer.memmap_path} ({size_mb:.1f} MB)")

    click.echo(f"\nSolve completed in {solve_time:.2f}s")
    click.echo(f"Result frames: {len(results)}")

    fmt_parts = [f.strip().lower() for f in output_format.split("+")]

    if "csv" in fmt_parts:
        if sample and pde_config.dimension == 2:
            click.echo("\nExporting sample slices (instead of full CSV)...")
            exporter = ResultExporter(pde_config, output_path)
            sample_output = exporter.export_sample(results, step_interval=step_interval)
            total_samples = sum(len(v) for v in sample_output.values())
            click.echo(f"  Exported {total_samples} sample files")
            for key, paths in sample_output.items():
                if paths:
                    click.echo(f"    {key}: {len(paths)} files")
        else:
            click.echo("\nExporting CSV files...")
            exporter = ResultExporter(pde_config, output_path)
            csv_paths = exporter.export_csv(results, step_interval=step_interval)
            click.echo(f"  Exported {len(csv_paths)} CSV files")

    if use_memmap and result_buffer is not None:
        pass
    elif "memmap" in fmt_parts:
        click.echo("\nSaving memmap data...")
        exporter = ResultExporter(pde_config, output_path)
        memmap_path = exporter.save_memmap(results)
        size_mb = memmap_path.stat().st_size / (1024 * 1024)
        click.echo(f"  Memmap saved: {memmap_path} ({size_mb:.1f} MB)")

    viz = Visualizer(pde_config, output_path)

    if "png" in fmt_parts:
        click.echo("\nGenerating PNG sequence...")
        png_paths = viz.generate_png_sequence(results, step_interval=step_interval)
        click.echo(f"  Generated {len(png_paths)} PNG frames")

    if "mp4" in fmt_parts:
        click.echo("\nGenerating MP4 animation...")
        mp4_path = viz.generate_mp4(
            results,
            fps=fps,
            step_interval=1,
            use_nvenc=use_nvenc,
        )
        if mp4_path:
            size_mb = mp4_path.stat().st_size / (1024 * 1024)
            click.echo(f"  Animation saved: {mp4_path} ({size_mb:.1f} MB)")
        else:
            click.echo("  WARNING: MP4 generation failed (ffmpeg may not be installed)")

    if use_memmap and result_buffer is not None:
        result_buffer.close()

    if summary:
        click.echo("\nExporting summary.json...")
        exporter = ResultExporter(pde_config, output_path)
        is_sparse = getattr(solver, "_use_sparse", False)
        memmap_size = 0.0
        if use_memmap and result_buffer is not None and result_buffer.memmap_path is not None:
            memmap_size = result_buffer.memmap_path.stat().st_size / (1024 * 1024)
        summary_path = exporter.export_summary(
            results,
            solve_time=solve_time,
            solver_name=solver_cls.__name__,
            use_memmap=use_memmap,
            memmap_size_mb=memmap_size,
            output_format=output_format,
            step_interval=step_interval,
            is_sparse=is_sparse,
        )
        click.echo(f"  Summary saved: {summary_path}")

    click.echo("\n" + "=" * 60)
    click.echo("  Done!")
    click.echo("=" * 60)


@cli.command()
@click.option(
    "--type",
    "-t",
    "pde_type",
    type=click.Choice(["heat", "wave", "poisson"]),
    required=True,
    help="PDE type to generate config for.",
)
@click.option(
    "--dim",
    type=click.Choice(["1", "2"]),
    default="2",
    help="Dimension (1 or 2).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output YAML file path.",
)
def init_config(pde_type: str, dim: str, output: str | None):
    """Generate a template YAML configuration file."""
    import yaml

    dimension = int(dim)
    config = _generate_template(pde_type, dimension)

    if output is None:
        output = f"{pde_type}_equation.yaml"

    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Template config saved to: {output}")


def _generate_template(pde_type: str, dimension: int) -> dict:
    config = {
        "pde_type": pde_type,
        "dimension": dimension,
    }

    if dimension == 1:
        config["domain_1d"] = {"x_min": 0.0, "x_max": 1.0, "nx": 100}
        config["boundary_conditions_1d"] = {
            "left": {"type": "dirichlet", "value": 0.0},
            "right": {"type": "dirichlet", "value": 0.0},
        }
    else:
        config["domain_2d"] = {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 100,
            "ny": 100,
        }
        config["boundary_conditions_2d"] = {
            "left": {"type": "dirichlet", "value": 0.0},
            "right": {"type": "dirichlet", "value": 0.0},
            "top": {"type": "dirichlet", "value": 0.0},
            "bottom": {"type": "dirichlet", "value": 0.0},
        }

    if pde_type == "heat":
        config["alpha"] = 1.0
        config["initial_condition"] = {
            "type": "function",
            "expression": "np.sin(np.pi * x)",
        }
        config["time"] = {"dt": 0.0001, "t_max": 0.01}
        config["solver"] = {"method": "auto", "max_iterations": 10000, "convergence_threshold": 1e-6}
    elif pde_type == "wave":
        config["c"] = 1.0
        config["initial_condition"] = {
            "type": "function",
            "expression": "np.sin(np.pi * x)",
        }
        config["initial_velocity"] = {"type": "constant", "value": 0.0}
        config["time"] = {"dt": 0.001, "t_max": 1.0}
        config["solver"] = {"method": "auto", "max_iterations": 10000, "convergence_threshold": 1e-6}
    elif pde_type == "poisson":
        config["source"] = "np.sin(np.pi * x) * np.sin(np.pi * y)" if dimension == 2 else "np.sin(np.pi * x)"
        config["solver"] = {"method": "gauss_seidel", "max_iterations": 10000, "convergence_threshold": 1e-6}

    return config


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file to validate.",
)
def validate(config: str):
    """Validate a YAML configuration file without solving."""
    try:
        pde_config = load_config(config)
        click.echo("Configuration is VALID")
        click.echo(f"  PDE Type: {pde_config.pde_type.value}")
        click.echo(f"  Dimension: {pde_config.dimension}D")
        if pde_config.time:
            click.echo(f"  Time steps: {pde_config.time.n_steps}")
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Configuration is INVALID: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
