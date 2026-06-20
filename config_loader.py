from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class PDEType(str, Enum):
    heat = "heat"
    wave = "wave"
    poisson = "poisson"


class BoundaryType(str, Enum):
    dirichlet = "dirichlet"
    neumann = "neumann"
    robin = "robin"


class BoundaryCondition(BaseModel):
    type: BoundaryType
    value: float = 0.0
    robin_alpha: Optional[float] = None
    robin_beta: Optional[float] = None

    @field_validator("robin_alpha", "robin_beta", mode="after")
    @classmethod
    def check_robin_params(cls, v: Optional[float], info) -> Optional[float]:
        return v


class BoundaryConditions1D(BaseModel):
    left: BoundaryCondition
    right: BoundaryCondition


class BoundaryConditions2D(BaseModel):
    left: BoundaryCondition
    right: BoundaryCondition
    top: BoundaryCondition
    bottom: BoundaryCondition


class Domain1D(BaseModel):
    x_min: float = 0.0
    x_max: float = 1.0
    nx: int = Field(ge=2)


class Domain2D(BaseModel):
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    nx: int = Field(ge=2)
    ny: int = Field(ge=2)


class TimeConfig(BaseModel):
    dt: float = Field(gt=0)
    t_max: float = Field(gt=0)
    n_steps: Optional[int] = None

    @model_validator(mode="after")
    def compute_n_steps(self) -> "TimeConfig":
        if self.n_steps is None:
            self.n_steps = int(self.t_max / self.dt)
        return self


class InitialCondition(BaseModel):
    type: str = "function"
    expression: Optional[str] = None
    value: Optional[float] = None


class SolverConfig(BaseModel):
    method: str = "auto"
    max_iterations: int = Field(default=10000, ge=1)
    convergence_threshold: float = Field(default=1e-6, gt=0)


class PDEConfig(BaseModel):
    pde_type: PDEType
    dimension: int = Field(ge=1, le=2)
    domain_1d: Optional[Domain1D] = None
    domain_2d: Optional[Domain2D] = None
    boundary_conditions_1d: Optional[BoundaryConditions1D] = None
    boundary_conditions_2d: Optional[BoundaryConditions2D] = None
    initial_condition: Optional[InitialCondition] = None
    initial_velocity: Optional[InitialCondition] = None
    time: Optional[TimeConfig] = None
    alpha: float = Field(default=1.0, gt=0)
    c: float = Field(default=1.0, gt=0)
    source: Optional[str] = None
    solver: SolverConfig = Field(default_factory=SolverConfig)

    @model_validator(mode="after")
    def check_consistency(self) -> "PDEConfig":
        if self.dimension == 1:
            if self.domain_1d is None:
                raise ValueError("domain_1d is required for 1D problems")
            if self.boundary_conditions_1d is None:
                raise ValueError("boundary_conditions_1d is required for 1D problems")
        elif self.dimension == 2:
            if self.domain_2d is None:
                raise ValueError("domain_2d is required for 2D problems")
            if self.boundary_conditions_2d is None:
                raise ValueError("boundary_conditions_2d is required for 2D problems")
        if self.pde_type in (PDEType.heat, PDEType.wave):
            if self.time is None:
                raise ValueError("time config is required for time-dependent PDEs")
        if self.pde_type == PDEType.wave:
            if self.initial_velocity is None:
                raise ValueError("initial_velocity is required for wave equation")
        return self

    @property
    def domain(self):
        if self.dimension == 1:
            return self.domain_1d
        return self.domain_2d

    @property
    def boundary_conditions(self):
        if self.dimension == 1:
            return self.boundary_conditions_1d
        return self.boundary_conditions_2d


def load_config(config_path: Union[str, Path]) -> PDEConfig:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        return PDEConfig(**raw)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
