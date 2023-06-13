#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path

FREQ = [148.0, 166.0, 201.0, 235.0, 283.0, 338.0, 388.0]
SBCEX17PATH = Path("../data/sbcex17")

@dataclass(frozen=True)
class SBCEX17Paths:
    path = SBCEX17PATH
    environment_data = SBCEX17PATH / "env_models" / "sbcex17.json"
    # gps_data = SBCEX17PATH / "gps" / "gps_range.csv"
    acoustic_path = SBCEX17PATH / "acoustic" / "processed"
    outputs = SBCEX17PATH / "outputs"
    ambiguity_surfaces = SBCEX17PATH / "acoustic" / "ambiguity_surfaces"
    # sbl_data = SBCEX17PATH / "acoustic" / "processed" / "SBL" / "results_peak.mat"
