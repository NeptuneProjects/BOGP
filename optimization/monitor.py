#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def print_lengthscale(client):
    try:
        try:
            try:
                print(client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["0"].lengthscale)
                print(client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["0"].raw_lengthscale)
                print(client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["1"].lengthscale)
                print(client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["1"].raw_lengthscale)
            except AttributeError:
                print(client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules["0"].lengthscale)
                print(client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules["0"].raw_lengthscale)
                print(client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules["1"].lengthscale)
                print(client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules["1"].raw_lengthscale)
        except AttributeError:
            print(f"num_outputs: {client._generation_strategy._model.model.model.num_outputs}")
            print(f"mean_module constant: {client._generation_strategy._model.model.model.mean_module.constant}")
            print(f"cov_module lengthscale: {client._generation_strategy._model.model.model.covar_module.base_kernel.lengthscale}")
            print(f"lengthscale prior: {client._generation_strategy._model.model.model.covar_module.base_kernel.lengthscale_prior}")
            print(f"prior concentration: {client._generation_strategy._model.model.model.covar_module.base_kernel.lengthscale_prior.concentration}")
            print(f"prior rate: {client._generation_strategy._model.model.model.covar_module.base_kernel.lengthscale_prior.rate}")
            print(f"output scale: {client._generation_strategy._model.model.model.covar_module.outputscale}")
            print(f"output scale prior: {client._generation_strategy._model.model.model.covar_module.outputscale_prior}")
            print(f"output scale prior: {client._generation_strategy._model.model.model.covar_module.outputscale_prior}")
            print(f"output scale prior concentration: {client._generation_strategy._model.model.model.covar_module.outputscale_prior.concentration}")
            print(f"output scale prior concentration: {client._generation_strategy._model.model.model.covar_module.outputscale_prior.rate}")
            print(f"raw output scale: {client._generation_strategy._model.model.model.covar_module.raw_outputscale}")
            print(f"raw output scale constraint: {client._generation_strategy._model.model.model.covar_module.raw_outputscale_constraint}")
            
        # print("Transforms:")
        # [print(t) for t in client._generation_strategy._model.transforms]
    except:
        print("Structure not found.")
        pass