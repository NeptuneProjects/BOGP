#!/usr/bin/env python3

import copy
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import random
from typing import Optional, Union
import warnings

from botorch import fit_gpytorch_model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import MCSampler
from botorch.test_functions import base
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.module import Module
import torch
from tqdm import tqdm

DTYPE = torch.double
logger = logging.getLogger(__name__)


class ObjectiveFunction:
    def __init__(self, obj_func):
        self.obj_func = obj_func

    def evaluate(self, parameters: dict, fixed_parameters: dict = {}):
        def _evaluate_objfunc():
            num_samples = set([len(param) for param in parameters.values()])
            if not len(num_samples) == 1:
                raise ValueError(
                    "The number of samples is inconsistent across features."
                )
            y = []
            for i in range(next(iter(num_samples))):
                params = {
                    k: float(v[i].detach().cpu().numpy()) for k, v in parameters.items()
                }
                y.append(self.obj_func(params | fixed_parameters))
            return torch.Tensor(y).unsqueeze(-1)

        def _evaluate_testfunc():
            X = torch.stack([v for v in parameters.values()], dim=1)
            return self.obj_func(X).unsqueeze(-1)

        try:
            if isinstance(self.obj_func, base.BaseTestProblem) or (
                "botorch" in self.obj_func.__module__
            ):
                return _evaluate_testfunc()
            else:
                return _evaluate_objfunc()
        except AttributeError:
            return _evaluate_objfunc()


class Optimizer:
    def __init__(
        self,
        obj_func: base.BaseTestProblem,
        search_parameters: dict,
        fixed_parameters: dict = {},
        obj_func_kwargs: dict = {},
        seed: int = 0,
    ):
        self.obj_func = ObjectiveFunction(obj_func)
        try:
            self.obj_func_name = obj_func._get_name()
        except AttributeError:
            self.obj_func_name = obj_func.__name__
        self.obj_func_module = obj_func.__module__
        self.search_parameters = search_parameters
        self.fixed_parameters = fixed_parameters
        self.obj_func_kwargs = obj_func_kwargs
        self.seed = seed
        self.bounds = self.get_bounds(self.search_parameters)
    
    @staticmethod
    def get_bounds(search_parameters):
        bounds = torch.zeros(2, len(search_parameters))
        for i, parameter in enumerate(search_parameters):
            bounds[:, i] = torch.tensor(parameter["bounds"])
        return bounds


@dataclass
class RandomSearchConfig:
    n_total: int = field(default=10)


class RandomSearch(Optimizer):
    def __init__(
        self,
        config: RandomSearchConfig,
        obj_func: base.BaseTestProblem,
        search_parameters: dict,
        fixed_parameters: dict = {},
        obj_func_kwargs: dict = {},
        seed: int = 0,
    ):
        super().__init__(
            obj_func,
            search_parameters,
            fixed_parameters,
            obj_func_kwargs,
            seed
        )
        self.config = config
        
    def get_candidates(self):
        """
        (r2 - r1) * torch.rand(M, N) + r1
        """
        torch.manual_seed(self.seed)
        # bounds = self.get_bounds(self.search_parameters)
        bounds = self.bounds
        N = bounds.shape[1]
        interval = (bounds[1] - bounds[0]).unsqueeze(-1)
        X_new = interval * torch.rand(N, self.config.n_total) + bounds[0].unsqueeze(-1)
        return X_new.T.detach().cpu()

    def run(self):
        X = self.get_candidates()
        y = []
        for Xi in X:
            parameters = {
                item["name"]: Xi[..., i].unsqueeze(-1) for i, item in enumerate(self.search_parameters)
            }
            y.append(self.obj_func.evaluate(parameters, self.fixed_parameters))
        
        return X, torch.tensor(y)


@dataclass
class BayesianOptimizationGPConfig:
    kernel_func: Optional[Module] = field(default=None)
    acq_func: AcquisitionFunction = field(default_factory=AcquisitionFunction)
    acq_func_kwargs: Union[dict, None] = field(default=None)
    sampler: Union[MCSampler, None] = field(default=None)
    sampler_kwargs: Union[dict, None] = field(default=None)
    batch_size: int = field(default=1)
    n_restarts: int = field(default=20)
    raw_samples: int = field(default=512)
    n_warmup: int = field(default=3)
    n_total: int = field(default=20)
    q: int = field(default=1)
    optimize_acqf_kwargs: Union[dict, None] = field(default=None)

    def __post_init__(self):
        self.n_iter = (
            self.n_total - self.n_warmup if self.n_total >= self.n_warmup else 0
        )
        self.acq_func_name = self.acq_func.__name__
        self.acq_func_module = self.acq_func.__module__
        if self.sampler is not None:
            self.sampler_name = self.sampler.__name__
            self.sampler_module = self.sampler.__module__
        else:
            self.sampler_name = None
            self.sampler_module = None

    def _construct_dict(self):
        return serialized_class_dict(self, {"acq_func", "sampler"})

    def __delete__(self):
        pass


class BayesianOptimizationGP(Optimizer):
    def __init__(
        self,
        config: BayesianOptimizationGPConfig,
        obj_func: base.BaseTestProblem,
        search_parameters: dict,
        fixed_parameters: dict = {},
        obj_func_kwargs: dict = {},
        device="cuda",
        seed: int = 0,
    ):
        super().__init__(
            obj_func,
            search_parameters,
            fixed_parameters,
            obj_func_kwargs,
            seed
        )
        self.config = config
        self.config_dict = config._construct_dict()
        # self.obj_func = ObjectiveFunction(obj_func)
        # try:
        #     self.obj_func_name = obj_func._get_name()
        # except AttributeError:
        #     self.obj_func_name = obj_func.__name__
        # self.obj_func_module = obj_func.__module__
        # self.search_parameters = search_parameters
        # self.fixed_parameters = fixed_parameters
        # self.obj_func_kwargs = obj_func_kwargs
        if (device == "cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.seed = seed
        # self.bounds = self.get_bounds(self.search_parameters)

    def __del__(self):
        pass

    def _construct_dict(self):
        return serialized_class_dict(self, {"config", "obj_func", "device"})

    # @staticmethod
    # def get_bounds(search_parameters):
    #     bounds = torch.zeros(2, len(search_parameters))
    #     for i, parameter in enumerate(search_parameters):
    #         bounds[:, i] = torch.tensor(parameter["bounds"])
    #     return bounds

    @staticmethod
    def initialize_model(X, y, state_dict=None, covar_module=None):
        model = SingleTaskGP(X, y, covar_module=covar_module).to(X)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model

    def run(self, disable_pbar=False):
        def _generate_initial_data():
            bounds = self.bounds
            X = (bounds[0] - bounds[1]) * torch.rand(
                self.config.n_warmup,
                bounds.size(1),
                dtype=DTYPE,
            ) + bounds[1]
            parameters = {
                item["name"]: X[..., i] for i, item in enumerate(self.search_parameters)
            }
            y = self.obj_func.evaluate(parameters, self.fixed_parameters)
            return X.to(self.device), y.to(self.device)

        def _get_acqfunc():
            if self.config.acq_func.__name__[0] == "q":
                acqfunc = self.config.acq_func(
                    model=model,
                    best_f=y.max(),
                    sampler=self.config.sampler(**self.config.sampler_kwargs),
                    **self.config.acq_func_kwargs,
                )
            else:
                acqfunc = self.config.acq_func(
                    model=model, best_f=y.max(), **self.config.acq_func_kwargs
                )
            return acqfunc

        def _optimize_acqf_and_get_observation(acq_func):
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds.to(self.device),
                q=self.config.q,
                num_restarts=self.config.n_restarts,
                raw_samples=self.config.raw_samples,
                options=self.config.optimize_acqf_kwargs,
            )
            X_new = candidates.detach()
            new_parameters = {
                item["name"]: X_new[..., i]
                for i, item in enumerate(self.search_parameters)
            }
            y_new = self.obj_func.evaluate(new_parameters, self.fixed_parameters)

            return X_new, y_new.to(self.device)

        try:

            warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            torch.manual_seed(self.seed)

            results = Results()

            X, y = _generate_initial_data()
            logger.info(f"Device: {self.device}")
            logger.info(
                f"Initial candidates: {list(map(tuple, X.detach().cpu().numpy()))}"
            )
            mll, model = self.initialize_model(
                X, y, covar_module=self.config.kernel_func
            )
            mll.to(self.device),
            model.to(self.device)

            fit_gpytorch_model(mll)

            pbar = tqdm(
                range(self.config.n_iter),
                desc="Optimizing",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                leave=True,
                position=0,
                unit=" eval",
                disable=disable_pbar,
                colour="blue",
            )

            count = 0
            for _ in pbar:
                acqfunc = _get_acqfunc()
                X_new, y_new = _optimize_acqf_and_get_observation(acqfunc)
                logger.info(
                    f"Iter {count} | New Candidates: {list(map(tuple, X_new.detach().cpu().numpy()))}"
                )
                with torch.no_grad():
                    saved_model = copy.deepcopy(model)
                    saved_acqfunc = copy.deepcopy(acqfunc)
                    results.append(
                        Result(
                            X.detach().cpu(),
                            y.detach().cpu(),
                            saved_model.to("cpu"),
                            saved_acqfunc.to("cpu"),
                            X_new.detach().cpu(),
                            y_new.detach().cpu(),
                        )
                    )

                X = torch.cat([X, X_new])
                y = torch.cat([y, y_new])

                mll, model = self.initialize_model(
                    X,
                    y,
                    state_dict=model.state_dict(),
                    covar_module=self.config.kernel_func,
                )

                fit_gpytorch_model(mll)
                if torch.cuda.is_available():
                    memory_free, memory_total = torch.cuda.mem_get_info(
                        self.device.type + ":0"
                    )
                    logger.info(
                        f"Iter {count} | CUDA Memory usage: {100 * (1 - memory_free / memory_total):.2f}%"
                    )
                count += 1

            with torch.no_grad():
                results.append(
                    Result(X.detach().cpu(), y.detach().cpu(), model.to("cpu"))
                )

        except:
            logger.exception("Exception raised in optimization loop.")
            raise Exception("Exception raised in optimization loop.")
        else:
            logger.info("Optimization completed.")
        finally:
            return results

    def save(self, path):
        torch.save(self._construct_dict(), path)


@dataclass
class Result:
    X: torch.Tensor = field(default_factory=torch.Tensor)
    y: torch.Tensor = field(default_factory=torch.Tensor)
    model: Model = field(default_factory=Model)
    acqfunc: Union[AcquisitionFunction, None] = field(default=None)
    X_new: Union[torch.Tensor, None] = field(default=None)
    y_new: Union[torch.Tensor, None] = field(default=None)

    def __post_init__(self):
        self.best_value = self.y.max().item()
        self.best_parameters = self.X[torch.argmax(self.y)]
        if self.acqfunc is not None:
            self.acqfunc_import = {
                "module": self.acqfunc.__module__,
                "name": self.acqfunc._get_name(),
            }
        else:
            self.acqfunc_import = None


class Results:
    def __init__(self, results: list = None):
        self.results = [] if results is None else results

    def __del__(self):
        pass

    def __getitem__(self, idx):
        return self.results[idx]

    def __len__(self):
        return len(self.results)

    def _construct_dict(self):
        results_dict = []
        for result in self.results:
            results_dict.append(
                {
                    "X": result.X,
                    "y": result.y,
                    "model": result.model.state_dict(),
                    "acqfunc": result.acqfunc_import,
                    "X_new": result.X_new,
                    "y_new": result.y_new,
                }
            )
        return results_dict

    def append(self, result):
        self.results.append(result)

    def load(self, path):
        results_dict = torch.load(path, map_location=torch.device("cpu"))

        for entry in results_dict:
            _, model = BayesianOptimizationGP.initialize_model(
                entry["X"], entry["y"], state_dict=entry["model"]
            )
            if entry["acqfunc"] is not None:
                AcqFunc = import_from_str(
                    entry["acqfunc"]["module"], entry["acqfunc"]["name"]
                )
                acqfunc = AcqFunc(model, best_f=entry["y"].max())
            else:
                acqfunc = None

            self.results.append(
                Result(
                    X=entry["X"],
                    y=entry["y"],
                    model=model,
                    acqfunc=acqfunc,
                    X_new=entry["X_new"],
                    y_new=entry["y_new"],
                )
            )
        return self.results

    def save(self, path):
        torch.save(self._construct_dict(), path)


def import_from_str(module: str, name: str):
    Obj = getattr(__import__(module, fromlist=[name]), name)
    return Obj


def run_mc(config_kwargs, optim_kwargs, path=Path.cwd(), num_trials=10, main_seed=0):

    random.seed(main_seed)
    trials = [random.randint(0, int(1e9)) for i in range(num_trials)]

    for trial in trials:

        trial_path = path / "Runs" / f"{trial:010d}"
        os.makedirs(trial_path, exist_ok=True)

        config = BayesianOptimizationGPConfig(**config_kwargs)
        optimizer = BayesianOptimizationGP(config, seed=trial, **optim_kwargs)
        results = optimizer.run()

        optimizer.save(trial_path / f"optim_{trial:010d}.pth")
        results.save(trial_path / f"results_{trial:010d}.pth")


def serialized_class_dict(cls, exclude_keys: dict):
    """Returns class dictionary excluding non-serializable objects."""
    return {k: v for k, v in cls.__dict__.items() if k not in exclude_keys}
