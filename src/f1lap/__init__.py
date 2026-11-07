"""f1lap: race lap-time modelling with validated uncertainty."""

from .data import COMPOUNDS, clean_laps, load_race, synthetic_race, validate_laps
from .models import HybridLapModel, PhysicsLapModel, QuantileGBM
from .openf1 import load_race_openf1

__version__ = "1.0.0"
__all__ = [
    "COMPOUNDS",
    "HybridLapModel",
    "PhysicsLapModel",
    "QuantileGBM",
    "clean_laps",
    "load_race",
    "load_race_openf1",
    "synthetic_race",
    "validate_laps",
]
