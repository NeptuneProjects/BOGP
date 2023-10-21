#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common

savepath = Path.cwd().parent / "reports/manuscripts/202310_JASA"

# sns.set_theme(style="white")

# df = sns.load_dataset("penguins")

# g = sns.PairGrid(df, diag_sharey=False)
# g.map_upper(sns.scatterplot, s=15)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.kdeplot, lw=2)
# plt.show()

def get_concat_h(im1, im2, im3):
    h_pad = 100
    dst = Image.new("RGB", (im1.width + im2.width + im3.width + 2 * h_pad, im2.height), "WHITE")
    dst.paste(im1, (0, 0)) 
    dst.paste(im2, (im1.width + h_pad, 0))
    dst.paste(im3, (im1.width + im2.width + 2 * h_pad, 0))
    return dst


def figure01():
    import sobol_demo
    sobol_demo.sample_comparison().savefig(savepath / "figure01.pdf", bbox_inches="tight", dpi=1000)


def figure03():
    images = [Image.open(x) for x in ["environment.png", "sensitivity_sim.png", "sensitivity_exp.png"]]
    get_concat_h(images[0], images[1], images[2]).save(savepath / "figure03.pdf", "PDF", resolution=1000)




if __name__ == "__main__":
    figure01()
    figure03()
