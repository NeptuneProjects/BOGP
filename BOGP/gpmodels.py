#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import Interval, Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class LocalizationGP(GPyTorchModel, ExactGP):
    _num_outputs = 1

    def __init__(self, train_X, train_Y) -> None:
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1])
        )
        self.to(train_X)

    def forward(self, train_X) -> MultivariateNormal:
        mean_x = self.mean_module(train_X)
        covar_x = self.covar_module(train_X)
        return MultivariateNormal(mean_x, covar_x)


class ConstrainedLocalizationGP(GPyTorchModel, ExactGP):
    _num_outputs = 1

    def __init__(self, train_X, train_Y) -> None:
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean(constant_constraint=Positive())

        self.covar_module = ScaleKernel(
            base_kernel=(
                RBFKernel(active_dims=[0], lengthscale_constraint=Interval(0.02, 0.2))
                * RBFKernel(active_dims=[1], lengthscale_constraint=Interval(0.02, 0.2))
            ),
        )
        self.to(train_X)

    def forward(self, train_X) -> MultivariateNormal:
        mean_x = self.mean_module(train_X)
        covar_x = self.covar_module(train_X)
        return MultivariateNormal(mean_x, covar_x)


def get_model_class(model_name: str) -> Union[GPyTorchModel, ExactGP]:
    """Helper function that implements the custom model classes."""
    if model_name == "LocalizationGP":
        return LocalizationGP
    elif model_name == "ConstrainedLocalizationGP":
        return ConstrainedLocalizationGP
    else:
        raise ValueError(f"Model {model_name} not implemented.")
