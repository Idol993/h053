import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config

print("Testing config with robin_beta...")
cfg = load_config("test_robin_beta_new.yaml")
bc = cfg.boundary_conditions_2d.left
print(f"  type: {bc.type}")
print(f"  value: {bc.value}")
print(f"  robin_alpha: {bc.robin_alpha}")
print(f"  robin_beta: {bc.robin_beta}")
print(f"  model_dump(): {bc.model_dump()}")

print()
print("Testing old config with value=beta...")
cfg2 = load_config("test_poisson_small_robin_left.yaml")
bc2 = cfg2.boundary_conditions_2d.left
print(f"  type: {bc2.type}")
print(f"  value: {bc2.value}")
print(f"  robin_alpha: {bc2.robin_alpha}")
print(f"  robin_beta: {bc2.robin_beta}")
print(f"  model_dump(): {bc2.model_dump()}")
