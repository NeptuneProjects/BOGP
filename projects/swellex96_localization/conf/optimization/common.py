#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path

FREQ = [148.0, 166.0, 201.0, 235.0, 283.0, 338.0, 388.0]
SWELLEX96PATH = Path("../data/swellex96_S5_VLA")


@dataclass(frozen=True)
class SWELLEX96Paths:
    path = SWELLEX96PATH
    environment_data = SWELLEX96PATH / "env_models" / "swellex96.json"
    gps_data = SWELLEX96PATH / "gps" / "gps_range.csv"
    acoustic_path = SWELLEX96PATH / "acoustic" / "processed"
    outputs = SWELLEX96PATH / "outputs"
    ambiguity_surfaces = SWELLEX96PATH / "acoustic" / "ambiguity_surfaces"
    sbl_data = SWELLEX96PATH / "acoustic" / "processed" / "SBL" / "results_peak.mat"
