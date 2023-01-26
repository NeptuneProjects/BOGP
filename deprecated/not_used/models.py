import pyro.distributions as dist
import pyro
import torch
import pdb

# Bayesian model
def waveguide(data):
    noise_std = data["noise_std"]  # Std noise
    m_dist = dist.Normal(data["m_mean"], data["m_std"])  # distribution of unknown parameter
    m = pyro.sample("m", m_dist)  # sample unknown parameter
    #phase_dist = dist.VonMises(data["phase_mean"], data["phase_concentration"])  # distribution of unknown parameter
    phase_dist = dist.Normal(data["phase_mean"], data["phase_std"])  # distribution of unknown parameter
    phase = pyro.sample("phase", phase_dist)  # sample unknown parameter
  #  print(phase)
    f_m = torch.sin(m * data["t"] + phase)  # non-linear function
    y_dist = dist.Normal(f_m, noise_std)  # likelihood
    with pyro.plate("Observations", size=data["obs"].size()[0]):
        y = pyro.sample("obs", y_dist, obs=data["obs"])  # sample likelihood


# Variational models
def variational_waveguide(data):
    # parameters to optimize over
    m_mean = pyro.param("m_mean", data["m_mean"])
    m_std = pyro.param("m_std",data["m_std"],constraint=dist.constraints.positive,)
    # Variational distribution
    m_dist = dist.Normal(m_mean, m_std)  # distribution of unknown parameter
    m = pyro.sample("m", m_dist)  # sample unknown parameter
    phase_mean = pyro.param("phase_mean", data["phase_guess"])
    phase_std  = pyro.param("phase_std", torch.tensor(1.0, device=data["device"]),
        constraint=dist.constraints.positive)
    # Variational distribution
    phase_dist = dist.Normal(phase_mean, phase_std)  # distribution of unknown parameter
    # phase_dist = dist.Delta(phase_mean)  # if you want the MAP
    phase = pyro.sample("phase", phase_dist)  # sample unknown parameter


# Bayesian model
def waveguide_1parm(data):
    noise_std = data["noise_std"]  # Std noise
    m_dist = dist.Normal(data["m_mean"], data["m_std"])  # distribution of unknown parameter
    m = pyro.sample("m", m_dist)  # sample unknown parameter
    f_m = torch.sin(m * data["t"])  # non-linear function
    y_dist = dist.Normal(f_m, noise_std)  # likelihood
    with pyro.plate("Observations", size=data["obs"].size()[0]):
        y = pyro.sample("obs", y_dist, obs=data["obs"])  # sample likelihood


# Variational models
def variational_waveguide_1parm(data):
    # parameters to optimize over
    m_mean = pyro.param("m_mean", data["m_mean"])
    m_std = pyro.param("m_std",data["m_std"], constraint=dist.constraints.positive, )
    # Variational distribution
    m_dist = dist.Normal(m_mean, m_std)  # distribution of unknown parameter
    m = pyro.sample("m", m_dist)  # sample unknown parameter
