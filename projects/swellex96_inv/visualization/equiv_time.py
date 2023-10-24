#!/usr/bin/env python
# -*- coding: utf-8 -*-

from string import ascii_lowercase
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots

plt.style.use(["science", "ieee", "std-colors"])


YLIM = (0.04, 1.0)
YTICKS = [0.05, 0.1, 0.5, 1.0]
MODES = ["Simulated", "Experimental"]