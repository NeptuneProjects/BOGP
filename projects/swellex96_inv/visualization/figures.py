#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

SAVEPATH = Path.cwd().parent / "reports/manuscripts/202401_JASA"
SAVE_KWARGS = {"bbox_inches": "tight", "dpi": 1200}


def figure01():
    import bo_examples

    bo_examples.main().savefig(SAVEPATH / "figure01.pdf", **SAVE_KWARGS)


def figure02():
    import environment

    environment.main().savefig(SAVEPATH / "figure02.pdf", **SAVE_KWARGS)


def figure03():
    import sensitivity

    sensitivity.main().savefig(SAVEPATH / "figure03.pdf", **SAVE_KWARGS)


def figure04():
    import diff_ev

    diff_ev.main().savefig(SAVEPATH / "figure04.pdf", **SAVE_KWARGS)


def figure05():
    import obj_performance

    obj_performance.main(n_init=64).savefig(SAVEPATH / "figure05.pdf", **SAVE_KWARGS)


def figure06():
    import param_est

    param_est.main(strategy="random", n_init=32).savefig(
        SAVEPATH / "figure06.pdf", **SAVE_KWARGS
    )


def figure07():
    import param_est

    param_est.main(strategy="ucb", n_init=64).savefig(
        SAVEPATH / "figure07.pdf", **SAVE_KWARGS
    )


def figure08():
    import warmup_perf

    warmup_perf.main(strategy="exp_logei").savefig(
        SAVEPATH / "figure08.pdf", **SAVE_KWARGS
    )


def figure09():
    import time_projection

    time_projection.main().savefig(SAVEPATH / "figure09.pdf", **SAVE_KWARGS)


if __name__ == "__main__":
    figure01()
    figure02()
    figure03()
    figure04()
    figure05()
    figure06()
    figure07()
    figure08()
    figure09()
