from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from config_loader import PDEConfig, PDEType


class ResultExporter:
    def __init__(self, config: PDEConfig, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, results: list[np.ndarray], step_interval: int = 1) -> list[Path]:
        paths = []
        if self.config.dimension == 1:
            paths = self._export_csv_1d(results, step_interval)
        else:
            paths = self._export_csv_2d(results, step_interval)
        return paths

    def _export_csv_1d(self, results: list[np.ndarray], step_interval: int) -> list[Path]:
        from mesh import Mesh1D

        paths = []
        for step, u in enumerate(results):
            if step % step_interval != 0:
                continue
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

    def _export_csv_2d(self, results: list[np.ndarray], step_interval: int) -> list[Path]:
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

        for step, u in enumerate(results):
            if step % step_interval != 0:
                continue
            filename = self.output_dir / f"result_t{step}.csv"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("x,y,u\n")
                for i in range(nx):
                    for j in range(ny):
                        f.write(f"{X[i, j]:.8f},{Y[i, j]:.8f},{u[i, j]:.8f}\n")
            paths.append(filename)
        return paths

    def save_memmap(self, results: list[np.ndarray]) -> Path:
        if not results:
            raise ValueError("No results to save")
        shape = (len(results),) + results[0].shape
        memmap_path = self.output_dir / "results.dat"
        mmap = np.memmap(
            str(memmap_path),
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )
        for i, u in enumerate(results):
            mmap[i] = u.astype(np.float32)
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
