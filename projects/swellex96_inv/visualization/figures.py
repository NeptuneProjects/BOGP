#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common

savepath = Path.cwd().parent / "reports/manuscripts/202401_JASA"

SAVE_KWARGS = {"bbox_inches": "tight", "dpi": 1000}


def get_concat_h(im1, im2, im3):
    h_pad = 100
    dst = Image.new(
        "RGB", (im1.width + im2.width + im3.width + 2 * h_pad, im2.height), "WHITE"
    )
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + h_pad, 0))
    dst.paste(im3, (im1.width + im2.width + 2 * h_pad, 0))
    return dst


def figure01():
    import bo_ucb_example, bo_ei_example, bo_logei_example

    bo_ucb_example.main().savefig(savepath / "figure01a.pdf", **SAVE_KWARGS)
    bo_ei_example.main().savefig(savepath / "figure01b.pdf", **SAVE_KWARGS)
    bo_logei_example.main().savefig(savepath / "figure01c.pdf", **SAVE_KWARGS)


def figure02():
    images = [
        Image.open(x)
        for x in ["environment.png", "sensitivity_sim.png", "sensitivity_exp.png"]
    ]
    get_concat_h(images[0], images[1], images[2]).save(
        savepath / "figure02.pdf", "PDF", resolution=1000
    )


def figure03():
    import obj_performance_rand

    obj_performance_rand.main(n_init=64).savefig(savepath / "figure03.pdf", **SAVE_KWARGS)

    # import obj_performance

    # obj_performance.main(n_init=64).savefig(savepath / "figure03.pdf", **SAVE_KWARGS)


def figure04():
    import diff_ev

    diff_ev.main().savefig(savepath / "figure04.pdf", **SAVE_KWARGS)


def figure05():
    import warmup_perf

    warmup_perf.main(strategy="exp_logei").savefig(
        savepath / "figure05.pdf", **SAVE_KWARGS
    )


def figure06():
    import param_est

    param_est.main(strategy="random", n_init=32).savefig(
        savepath / "figure06.pdf", **SAVE_KWARGS
    )


def figure07():
    import param_est

    param_est.main(strategy="ucb", n_init=64).savefig(
        savepath / "figure07.pdf", **SAVE_KWARGS
    )


def figure08():
    # import time_projection

    # time_projection.main().savefig(savepath / "figure08.pdf", **SAVE_KWARGS)
    import time_projection_rand

    time_projection_rand.main().savefig(savepath / "figure08.pdf", **SAVE_KWARGS)


if __name__ == "__main__":
    # figure01()
    # figure02()
    # figure03()
    # figure04()
    # figure05()
    # figure06()
    # figure07()
    figure08()
