"""floway: a lightweight flow cytometry analysis toolkit."""

from .parser import FCSParser
from .gating import (PlateGate, analyze_fcs_data, gate_populations,
                     gate_singlets_mahalanobis)
from .plotting import plot_gating_strategy, plot_kde, plot_kde_grid
from .utils import parse_plate_map
from .presets import PRESETS, resolve_channel
from .diagnostics import channel_activity
from .auto import auto_thresholds
from .channels import detect_channels, find_channel

__version__ = "0.1.0"

__all__ = [
    "FCSParser",
    "gate_populations",
    "PlateGate",
    "gate_singlets_mahalanobis",
    "analyze_fcs_data",
    "auto_thresholds",
    "plot_gating_strategy",
    "plot_kde",
    "plot_kde_grid",
    "parse_plate_map",
    "PRESETS",
    "resolve_channel",
    "detect_channels",
    "find_channel",
    "channel_activity",
]
