from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Union

import numpy as np

from config_loader import PDEConfig, PDEType


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
