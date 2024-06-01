#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

from ax.plot.contour import interact_contour
from ax.plot.slice import plot_slice
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import render

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common


def main():
    client = AxClient.load_from_json_file(common.SWELLEX96Paths.outputs / "client.json")
    client.fit_model()
    render(client.get_optimization_trace(objective_optimum=0.0))

    model = client.generation_strategy.model
    render(interact_contour(model=model, metric_name="bartlett"))
    render(plot_slice(model, "rec_r", "bartlett"))
    render(plot_slice(model, "src_z", "bartlett"))
    render(plot_slice(model, "h_w", "bartlett"))
    render(plot_slice(model, "c_s", "bartlett"))
    render(plot_slice(model, "dcdz_s", "bartlett"))
    render(plot_slice(model, "tilt", "bartlett"))

    print(client.get_best_parameters(use_model_predictions=False))


if __name__ == "__main__":
    main()
