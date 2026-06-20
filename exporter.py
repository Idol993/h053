from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Union

import numpy as np

from config_loader import PDEConfig, PDEType, BoundaryType


class ResultBuffer:
    def __init__(
        self,
        frame_shape: tuple,
        n_frames: int,
        use_memmap: bool = False,
        memmap_path: str | Path | None = None,
        dtype: type = np.float64,
    ):
        self.frame_shape = frame_shape
        self.n_frames = n_frames
        self.use_memmap = use_memmap
        self.dtype = dtype
        self._count = 0
        self._memmap_path = Path(memmap_path) if memmap_path else None

        if use_memmap:
            if not self._memmap_path:
                raise ValueError("memmap_path must be provided when use_memmap=True")
            self._memmap_path.parent.mkdir(parents=True, exist_ok=True)
            self._data = np.memmap(
                str(self._memmap_path),
                dtype=dtype,
                mode="w+",
                shape=(n_frames,) + frame_shape,
            )
        else:
            self._data: list[np.ndarray] = []

    def append(self, frame: np.ndarray):
        if self._count >= self.n_frames:
            raise IndexError(
                f"ResultBuffer is full: {self._count}/{self.n_frames} frames"
            )
        if self.use_memmap:
            self._data[self._count] = frame.astype(self.dtype)
        else:
            self._data.append(frame.copy())
        self._count += 1

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx = self._count + idx
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Index {idx} out of range (0..{self._count - 1})")
        if self.use_memmap:
            return self._data[idx].copy()
        return self._data[idx]

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self._count):
            yield self[i]

    @property
    def full_shape(self) -> tuple:
        return (self._count,) + self.frame_shape

    def flush(self):
        if self.use_memmap:
            self._data.flush()

    def close(self):
        if self.use_memmap:
            self._data.flush()
            del self._data

    @property
    def memmap_path(self) -> Path | None:
        return self._memmap_path

    @staticmethod
    def from_config(
        config: PDEConfig,
        output_dir: str | Path,
        use_memmap: bool = False,
    ) -> "ResultBuffer":
        if config.dimension == 1:
            frame_shape = (config.domain_1d.nx,)
        else:
            frame_shape = (config.domain_2d.nx, config.domain_2d.ny)

        if config.pde_type == PDEType.poisson:
            n_frames = 1
        else:
            n_frames = config.time.n_steps + 1

        if use_memmap:
            memmap_path = Path(output_dir) / "results.dat"
            return ResultBuffer(
                frame_shape=frame_shape,
                n_frames=n_frames,
                use_memmap=True,
                memmap_path=memmap_path,
                dtype=np.float32,
            )
        return ResultBuffer(
            frame_shape=frame_shape,
            n_frames=n_frames,
            use_memmap=False,
            dtype=np.float64,
        )

    def to_list(self) -> list[np.ndarray]:
        return [self[i] for i in range(len(self))]


class ResultExporter:
    def __init__(self, config: PDEConfig, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, results: Union[ResultBuffer, list[np.ndarray]], step_interval: int = 1) -> list[Path]:
        paths = []
        if self.config.dimension == 1:
            paths = self._export_csv_1d(results, step_interval)
        else:
            paths = self._export_csv_2d(results, step_interval)
        return paths

    def _export_csv_1d(
        self, results: Union[ResultBuffer, list[np.ndarray]], step_interval: int
    ) -> list[Path]:
        paths = []
        for step in range(0, len(results), step_interval):
            u = results[step]
            filename = self.output_dir / f"result_t{step}.csv"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("x,u\n")
                for i in range(len(u)):
                    x_val = self.config.domain_1d.x_min + i * (
                        (self.config.domain_1d.x_max - self.config.domain_1d.x_min)
                        / (self.config.domain_1d.nx - 1)
                    )
                    f.write(f"{x_val:.8f},{u[i]:.8f}\n")
            paths.append(filename)
        return paths

    def _export_csv_2d(
        self, results: Union[ResultBuffer, list[np.ndarray]], step_interval: int
    ) -> list[Path]:
        paths = []
        nx = self.config.domain_2d.nx
        ny = self.config.domain_2d.ny
        x = np.linspace(
            self.config.domain_2d.x_min,
            self.config.domain_2d.x_max,
            nx,
        )
        y = np.linspace(
            self.config.domain_2d.y_min,
            self.config.domain_2d.y_max,
            ny,
        )
        X, Y = np.meshgrid(x, y, indexing="ij")

        for step in range(0, len(results), step_interval):
            u = results[step]
            filename = self.output_dir / f"result_t{step}.csv"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("x,y,u\n")
                for i in range(nx):
                    for j in range(ny):
                        f.write(f"{X[i, j]:.8f},{Y[i, j]:.8f},{u[i, j]:.8f}\n")
            paths.append(filename)
        return paths

    def save_memmap(self, results: Union[ResultBuffer, list[np.ndarray]]) -> Path:
        if len(results) == 0:
            raise ValueError("No results to save")
        shape = (len(results),) + results[0].shape
        memmap_path = self.output_dir / "results.dat"
        mmap = np.memmap(
            str(memmap_path),
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )
        for i in range(len(results)):
            mmap[i] = results[i].astype(np.float32)
        del mmap
        meta_path = self.output_dir / "results_meta.npz"
        np.savez(
            str(meta_path),
            shape=np.array(shape),
            dtype=np.array(["float32"]),
            n_steps=np.array([len(results)]),
        )
        return memmap_path

    @staticmethod
    def load_memmap(memmap_path: str | Path, shape: tuple) -> np.ndarray:
        return np.memmap(str(memmap_path), dtype=np.float32, mode="r", shape=shape)

    def export_summary(
        self,
        results: Union[ResultBuffer, list[np.ndarray]],
        solve_time: float = 0.0,
        solver_name: str = "",
        use_memmap: bool = False,
        memmap_size_mb: float = 0.0,
        output_format: str = "",
        step_interval: int = 1,
        is_sparse: bool = False,
        sampling_enabled: bool = False,
        original_grid_shape: tuple | None = None,
        n_sample_files: int = 0,
    ) -> Path:
        summary: dict = {}

        summary["pde_type"] = self.config.pde_type.value
        summary["dimension"] = self.config.dimension

        if self.config.dimension == 1:
            summary["mesh"] = {
                "nx": self.config.domain_1d.nx,
                "dx": float(self.config.domain_1d.x_max - self.config.domain_1d.x_min) / (self.config.domain_1d.nx - 1),
                "x_min": self.config.domain_1d.x_min,
                "x_max": self.config.domain_1d.x_max,
            }
        else:
            dx = float(self.config.domain_2d.x_max - self.config.domain_2d.x_min) / (self.config.domain_2d.nx - 1)
            dy = float(self.config.domain_2d.y_max - self.config.domain_2d.y_min) / (self.config.domain_2d.ny - 1)
            summary["mesh"] = {
                "nx": self.config.domain_2d.nx,
                "ny": self.config.domain_2d.ny,
                "dx": dx,
                "dy": dy,
                "x_min": self.config.domain_2d.x_min,
                "x_max": self.config.domain_2d.x_max,
                "y_min": self.config.domain_2d.y_min,
                "y_max": self.config.domain_2d.y_max,
            }

        if self.config.time is not None:
            summary["time"] = {
                "dt": self.config.time.dt,
                "t_max": self.config.time.t_max,
                "n_steps": self.config.time.n_steps,
            }
        else:
            summary["time"] = None

        summary["solver"] = {
            "method": self.config.solver.method if self.config.solver else "",
            "name": solver_name,
            "is_sparse": is_sparse,
            "solve_time_seconds": solve_time,
        }

        summary["memmap"] = {
            "enabled": use_memmap,
            "size_mb": memmap_size_mb,
        }

        n_exported = (len(results) + step_interval - 1) // step_interval if step_interval > 0 else 0
        summary["export"] = {
            "format": output_format,
            "n_frames_total": len(results),
            "n_frames_exported": n_exported,
            "step_interval": step_interval,
            "sampling_enabled": sampling_enabled,
            "n_sample_files": n_sample_files,
        }

        if sampling_enabled and original_grid_shape is not None:
            summary["mesh"]["original_grid_shape"] = list(original_grid_shape)

        if len(results) > 0:
            last_frame = results[-1]
            summary["stats"] = {
                "min": float(np.min(last_frame)),
                "max": float(np.max(last_frame)),
                "mean": float(np.mean(last_frame)),
                "std": float(np.std(last_frame)),
            }
            summary["boundary_residuals"] = self._compute_boundary_residuals(last_frame)
        else:
            summary["stats"] = {}
            summary["boundary_residuals"] = {}

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary_path

    def _compute_boundary_residuals(self, u: np.ndarray) -> dict:
        residuals: dict = {}

        def _residual_stats(arr: np.ndarray, exclude_corners: bool = False) -> dict:
            result = {
                "max_abs": float(np.max(np.abs(arr))),
                "rms": float(np.sqrt(np.mean(arr ** 2))),
                "mean": float(np.mean(arr)),
            }
            if exclude_corners and len(arr) > 2:
                interior = arr[1:-1]
                result["interior_max_abs"] = float(np.max(np.abs(interior)))
                result["interior_rms"] = float(np.sqrt(np.mean(interior ** 2)))
                result["interior_mean"] = float(np.mean(interior))
            return result

        if self.config.dimension == 1:
            bc = self.config.boundary_conditions_1d
            dx = (self.config.domain_1d.x_max - self.config.domain_1d.x_min) / (self.config.domain_1d.nx - 1)

            left_type = bc.left.type.value
            if bc.left.type == BoundaryType.dirichlet:
                res = np.array([u[0] - bc.left.value])
            elif bc.left.type == BoundaryType.neumann:
                dudx = (u[0] - u[1]) / dx
                res = np.array([dudx - bc.left.value])
            elif bc.left.type == BoundaryType.robin:
                alpha = bc.left.robin_alpha if bc.left.robin_alpha is not None else 0.0
                beta = bc.left.robin_beta if bc.left.robin_beta is not None else bc.left.value
                dudn = (u[0] - u[1]) / dx
                res = np.array([alpha * u[0] + dudn - beta])
            else:
                res = np.array([0.0])
            residuals["left"] = {"type": left_type, **_residual_stats(res)}

            right_type = bc.right.type.value
            if bc.right.type == BoundaryType.dirichlet:
                res = np.array([u[-1] - bc.right.value])
            elif bc.right.type == BoundaryType.neumann:
                dudx = (u[-1] - u[-2]) / dx
                res = np.array([dudx - bc.right.value])
            elif bc.right.type == BoundaryType.robin:
                alpha = bc.right.robin_alpha if bc.right.robin_alpha is not None else 0.0
                beta = bc.right.robin_beta if bc.right.robin_beta is not None else bc.right.value
                dudn = (u[-1] - u[-2]) / dx
                res = np.array([alpha * u[-1] + dudn - beta])
            else:
                res = np.array([0.0])
            residuals["right"] = {"type": right_type, **_residual_stats(res)}

        else:
            bc = self.config.boundary_conditions_2d
            dx = (self.config.domain_2d.x_max - self.config.domain_2d.x_min) / (self.config.domain_2d.nx - 1)
            dy = (self.config.domain_2d.y_max - self.config.domain_2d.y_min) / (self.config.domain_2d.ny - 1)

            left_type = bc.left.type.value
            if bc.left.type == BoundaryType.dirichlet:
                res = u[0, :] - bc.left.value
            elif bc.left.type == BoundaryType.neumann:
                dudx = (u[0, :] - u[1, :]) / dx
                res = dudx - bc.left.value
            elif bc.left.type == BoundaryType.robin:
                alpha = bc.left.robin_alpha if bc.left.robin_alpha is not None else 0.0
                beta = bc.left.robin_beta if bc.left.robin_beta is not None else bc.left.value
                dudn = (u[0, :] - u[1, :]) / dx
                res = alpha * u[0, :] + dudn - beta
            else:
                res = np.zeros_like(u[0, :])
            residuals["left"] = {"type": left_type, **_residual_stats(res, exclude_corners=True)}

            right_type = bc.right.type.value
            if bc.right.type == BoundaryType.dirichlet:
                res = u[-1, :] - bc.right.value
            elif bc.right.type == BoundaryType.neumann:
                dudx = (u[-1, :] - u[-2, :]) / dx
                res = dudx - bc.right.value
            elif bc.right.type == BoundaryType.robin:
                alpha = bc.right.robin_alpha if bc.right.robin_alpha is not None else 0.0
                beta = bc.right.robin_beta if bc.right.robin_beta is not None else bc.right.value
                dudn = (u[-1, :] - u[-2, :]) / dx
                res = alpha * u[-1, :] + dudn - beta
            else:
                res = np.zeros_like(u[-1, :])
            residuals["right"] = {"type": right_type, **_residual_stats(res, exclude_corners=True)}

            bottom_type = bc.bottom.type.value
            if bc.bottom.type == BoundaryType.dirichlet:
                res = u[:, 0] - bc.bottom.value
            elif bc.bottom.type == BoundaryType.neumann:
                dudy = (u[:, 0] - u[:, 1]) / dy
                res = dudy - bc.bottom.value
            elif bc.bottom.type == BoundaryType.robin:
                alpha = bc.bottom.robin_alpha if bc.bottom.robin_alpha is not None else 0.0
                beta = bc.bottom.robin_beta if bc.bottom.robin_beta is not None else bc.bottom.value
                dudn = (u[:, 0] - u[:, 1]) / dy
                res = alpha * u[:, 0] + dudn - beta
            else:
                res = np.zeros_like(u[:, 0])
            residuals["bottom"] = {"type": bottom_type, **_residual_stats(res, exclude_corners=True)}

            top_type = bc.top.type.value
            if bc.top.type == BoundaryType.dirichlet:
                res = u[:, -1] - bc.top.value
            elif bc.top.type == BoundaryType.neumann:
                dudy = (u[:, -1] - u[:, -2]) / dy
                res = dudy - bc.top.value
            elif bc.top.type == BoundaryType.robin:
                alpha = bc.top.robin_alpha if bc.top.robin_alpha is not None else 0.0
                beta = bc.top.robin_beta if bc.top.robin_beta is not None else bc.top.value
                dudn = (u[:, -1] - u[:, -2]) / dy
                res = alpha * u[:, -1] + dudn - beta
            else:
                res = np.zeros_like(u[:, -1])
            residuals["top"] = {"type": top_type, **_residual_stats(res, exclude_corners=True)}

        return residuals

    def export_sample(
        self,
        results: Union[ResultBuffer, list[np.ndarray]],
        step_interval: int = 1,
        n_boundary_layers: int = 2,
    ) -> dict:
        output: dict[str, list[Path]] = {
            "centerline_x": [],
            "centerline_y": [],
            "boundary_left": [],
            "boundary_right": [],
            "boundary_bottom": [],
            "boundary_top": [],
            "stats": [],
        }

        if self.config.dimension == 1:
            return output

        nx = self.config.domain_2d.nx
        ny = self.config.domain_2d.ny
        x = np.linspace(self.config.domain_2d.x_min, self.config.domain_2d.x_max, nx)
        y = np.linspace(self.config.domain_2d.y_min, self.config.domain_2d.y_max, ny)

        mid_x = nx // 2
        mid_y = ny // 2

        for step in range(0, len(results), step_interval):
            u = results[step]
            tag = f"t{step}"

            cl_x_path = self.output_dir / f"sample_centerline_x_{tag}.csv"
            with open(cl_x_path, "w", encoding="utf-8") as f:
                f.write("x,u\n")
                for i in range(nx):
                    f.write(f"{x[i]:.8f},{u[i, mid_y]:.8f}\n")
            output["centerline_x"].append(cl_x_path)

            cl_y_path = self.output_dir / f"sample_centerline_y_{tag}.csv"
            with open(cl_y_path, "w", encoding="utf-8") as f:
                f.write("y,u\n")
                for j in range(ny):
                    f.write(f"{y[j]:.8f},{u[mid_x, j]:.8f}\n")
            output["centerline_y"].append(cl_y_path)

            bl_path = self.output_dir / f"sample_boundary_left_{tag}.csv"
            with open(bl_path, "w", encoding="utf-8") as f:
                cols = ["y"] + [f"u_i{i}" for i in range(min(n_boundary_layers, nx))]
                f.write(",".join(cols) + "\n")
                for j in range(ny):
                    vals = [f"{y[j]:.8f}"] + [f"{u[i, j]:.8f}" for i in range(min(n_boundary_layers, nx))]
                    f.write(",".join(vals) + "\n")
            output["boundary_left"].append(bl_path)

            br_path = self.output_dir / f"sample_boundary_right_{tag}.csv"
            with open(br_path, "w", encoding="utf-8") as f:
                cols = ["y"] + [f"u_i{nx-1-i}" for i in range(min(n_boundary_layers, nx))]
                f.write(",".join(cols) + "\n")
                for j in range(ny):
                    vals = [f"{y[j]:.8f}"] + [f"{u[nx-1-i, j]:.8f}" for i in range(min(n_boundary_layers, nx))]
                    f.write(",".join(vals) + "\n")
            output["boundary_right"].append(br_path)

            bb_path = self.output_dir / f"sample_boundary_bottom_{tag}.csv"
            with open(bb_path, "w", encoding="utf-8") as f:
                cols = ["x"] + [f"u_j{j}" for j in range(min(n_boundary_layers, ny))]
                f.write(",".join(cols) + "\n")
                for i in range(nx):
                    vals = [f"{x[i]:.8f}"] + [f"{u[i, j]:.8f}" for j in range(min(n_boundary_layers, ny))]
                    f.write(",".join(vals) + "\n")
            output["boundary_bottom"].append(bb_path)

            bt_path = self.output_dir / f"sample_boundary_top_{tag}.csv"
            with open(bt_path, "w", encoding="utf-8") as f:
                cols = ["x"] + [f"u_j{ny-1-j}" for j in range(min(n_boundary_layers, ny))]
                f.write(",".join(cols) + "\n")
                for i in range(nx):
                    vals = [f"{x[i]:.8f}"] + [f"{u[i, ny-1-j]:.8f}" for j in range(min(n_boundary_layers, ny))]
                    f.write(",".join(vals) + "\n")
            output["boundary_top"].append(bt_path)

            stats_path = self.output_dir / f"sample_stats_{tag}.json"
            stats = {
                "step": step,
                "min": float(np.min(u)),
                "max": float(np.max(u)),
                "mean": float(np.mean(u)),
                "std": float(np.std(u)),
                "l2_norm": float(np.sqrt(np.mean(u ** 2))),
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            output["stats"].append(stats_path)

        return output
