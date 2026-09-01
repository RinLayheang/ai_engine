"""Training/inference configuration. Defaults match Document/PROPOSAL.md item 5."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = "xlm-roberta-base"
    max_length: int = 160
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 4
    seed: int = 42

    # Per-axis abstention thresholds. A verdict abstains if either axis's
    # confidence falls below its threshold. Calibrated on held-out (eval)
    # data by evaluate.calibrate_thresholds — these are placeholder defaults
    # until that calibration has real data to run on.
    severity_threshold: float = 0.6
    target_threshold: float = 0.6

    model_version: str = "v0-untrained"
