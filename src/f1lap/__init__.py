"""f1lap: race lap-time modelling with validated uncertainty."""

from .data import COMPOUNDS, clean_laps, load_race, synthetic_race, validate_laps
from .models import HybridLapModel, PhysicsLapModel, QuantileGBM

__version__ = "1.0.0"
__all__ = [
    "COMPOUNDS",
    "HybridLapModel",
    "PhysicsLapModel",
    "QuantileGBM",
    "clean_laps",
    "load_race",
    "synthetic_race",
    "validate_laps",
]
