from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

from config_loader import PDEConfig, PDEType

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, config: PDEConfig, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def generate_png_sequence(
        self,
        results: list[np.ndarray],
        step_interval: int = 10,
        dpi: int = 150,
    ) -> list[Path]:
        paths = []
        if self.config.dimension == 1:
            paths = self._generate_png_1d(results, step_interval, dpi)
        else:
            paths = self._generate_png_2d(results, step_interval, dpi)
        return paths

    def _generate_png_1d(
        self, results: list[np.ndarray], step_interval: int, dpi: int
    ) -> list[Path]:
        all_vals = np.concatenate([r.flatten() for r in results])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        x = np.linspace(
            self.config.domain_1d.x_min,
            self.config.domain_1d.x_max,
            self.config.domain_1d.nx,
        )
        paths = []
        for step, u in enumerate(results):
            if step % step_interval != 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, u, "b-", linewidth=1.5)
            ax.set_xlim(self.config.domain_1d.x_min, self.config.domain_1d.x_max)
            ax.set_ylim(vmin - 0.1 * abs(vmin) - 0.01, vmax + 0.1 * abs(vmax) + 0.01)
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            pde_name = self.config.pde_type.value
            ax.set_title(f"{pde_name} equation - t={step * self.config.time.dt:.4f}")
            ax.grid(True, alpha=0.3)
            filepath = self.frames_dir / f"frame_{step:06d}.png"
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(filepath)
        return paths

    def _generate_png_2d(
        self, results: list[np.ndarray], step_interval: int, dpi: int
    ) -> list[Path]:
        all_vals = np.concatenate([r.flatten() for r in results])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        nx = self.config.domain_2d.nx
        ny = self.config.domain_2d.ny
        x = np.linspace(self.config.domain_2d.x_min, self.config.domain_2d.x_max, nx)
        y = np.linspace(self.config.domain_2d.y_min, self.config.domain_2d.y_max, ny)
        paths = []
        for step, u in enumerate(results):
            if step % step_interval != 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.pcolormesh(x, y, u.T, cmap="hot", norm=norm, shading="auto")
            fig.colorbar(im, ax=ax, label="u")
            pde_name = self.config.pde_type.value
            dt = self.config.time.dt if self.config.time else 0
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{pde_name} equation - t={step * dt:.4f}")
            filepath = self.frames_dir / f"frame_{step:06d}.png"
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(filepath)
        return paths

    def generate_mp4(
        self,
        results: list[np.ndarray],
        fps: int = 10,
        resolution: tuple[int, int] = (1920, 1080),
        step_interval: int = 1,
        codec: str = "ffmpeg",
        use_nvenc: bool = False,
    ) -> Optional[Path]:
        if self.config.dimension == 1:
            return self._generate_mp4_1d(results, fps, resolution, step_interval, codec, use_nvenc)
        return self._generate_mp4_2d(results, fps, resolution, step_interval, codec, use_nvenc)

    def _generate_mp4_1d(
        self, results, fps, resolution, step_interval, codec, use_nvenc
    ) -> Optional[Path]:
        frames_data = []
        for step, u in enumerate(results):
            if step % step_interval != 0:
                continue
            frames_data.append((step, u))
        if not frames_data:
            return None

        all_vals = np.concatenate([u.flatten() for _, u in frames_data])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        x = np.linspace(
            self.config.domain_1d.x_min,
            self.config.domain_1d.x_max,
            self.config.domain_1d.nx,
        )

        fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100))
        (line,) = ax.plot(x, frames_data[0][1], "b-", linewidth=1.5)
        ax.set_xlim(self.config.domain_1d.x_min, self.config.domain_1d.x_max)
        ax.set_ylim(vmin - 0.1 * abs(vmin) - 0.01, vmax + 0.1 * abs(vmax) + 0.01)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        pde_name = self.config.pde_type.value
        title = ax.set_title(f"{pde_name} equation - t=0.0000")
        ax.grid(True, alpha=0.3)

        def update(frame_idx):
            step, u = frames_data[frame_idx]
            line.set_ydata(u)
            dt = self.config.time.dt if self.config.time else 0
            title.set_text(f"{pde_name} equation - t={step * dt:.4f}")
            return line, title

        ani = animation.FuncAnimation(
            fig, update, frames=len(frames_data), interval=1000 // fps, blit=False
        )

        output_path = self.output_dir / "animation.mp4"
        writer_kwargs = {"fps": fps, "dpi": 100}
        if use_nvenc:
            writer_kwargs["extra_args"] = ["-c:v", "h264_nvenc"]
        try:
            writer = animation.FFMpegWriter(**writer_kwargs)
            ani.save(str(output_path), writer=writer)
        except Exception as e:
            logger.warning(f"FFMpeg MP4 generation failed: {e}. Trying pillow fallback...")
            try:
                gif_path = self.output_dir / "animation.gif"
                ani.save(str(gif_path), writer="pillow", fps=fps)
                plt.close(fig)
                return gif_path
            except Exception as e2:
                logger.error(f"Animation generation failed completely: {e2}")
                plt.close(fig)
                return None
        plt.close(fig)
        return output_path

    def _generate_mp4_2d(
        self, results, fps, resolution, step_interval, codec, use_nvenc
    ) -> Optional[Path]:
        frames_data = []
        for step, u in enumerate(results):
            if step % step_interval != 0:
                continue
            frames_data.append((step, u))
        if not frames_data:
            return None

        all_vals = np.concatenate([u.flatten() for _, u in frames_data])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        nx = self.config.domain_2d.nx
        ny = self.config.domain_2d.ny
        x = np.linspace(self.config.domain_2d.x_min, self.config.domain_2d.x_max, nx)
        y = np.linspace(self.config.domain_2d.y_min, self.config.domain_2d.y_max, ny)

        fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100))
        im = ax.pcolormesh(x, y, frames_data[0][1].T, cmap="hot", norm=norm, shading="auto")
        fig.colorbar(im, ax=ax, label="u")
        pde_name = self.config.pde_type.value
        title = ax.set_title(f"{pde_name} equation - t=0.0000")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def update(frame_idx):
            step, u = frames_data[frame_idx]
            im.set_array(u.T.ravel())
            dt = self.config.time.dt if self.config.time else 0
            title.set_text(f"{pde_name} equation - t={step * dt:.4f}")
            return im, title

        ani = animation.FuncAnimation(
            fig, update, frames=len(frames_data), interval=1000 // fps, blit=False
        )

        output_path = self.output_dir / "animation.mp4"
        writer_kwargs = {"fps": fps, "dpi": 100}
        if use_nvenc:
            writer_kwargs["extra_args"] = ["-c:v", "h264_nvenc"]
        try:
            writer = animation.FFMpegWriter(**writer_kwargs)
            ani.save(str(output_path), writer=writer)
        except Exception as e:
            logger.warning(f"FFMpeg MP4 generation failed: {e}. Trying pillow fallback...")
            try:
                gif_path = self.output_dir / "animation.gif"
                ani.save(str(gif_path), writer="pillow", fps=fps)
                plt.close(fig)
                return gif_path
            except Exception as e2:
                logger.error(f"Animation generation failed completely: {e2}")
                plt.close(fig)
                return None
        plt.close(fig)
        return output_path
