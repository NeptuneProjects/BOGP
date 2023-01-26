#%% Libraries
import numpy as np
from tqdm import tqdm
import pyro
import time
import torch
from pyro.infer import Trace_ELBO, SVI
import pdb
import matplotlib.pyplot as plt
import models as models  # This one is the file with the models. Not needed to install

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%% Simulated data
t = np.linspace(0, 1, 100)
m = 10  # Value to find
m_guess = 20.0  # Initial guessed value (prior)
phase = np.pi
phase_guess = 4.0  # phase
y = np.sin(m * t + phase)  # real field
y_guess = np.sin(m_guess * t + phase_guess)  # guessed field
noise_std = 0.9  # noise
data = y + noise_std * np.random.randn(len(t))  # measured field
plt.plot(t, y)
plt.plot(t, data)
plt.plot(t, y_guess)
plt.legend(labels=["True", "Measured", "Initial guess"])
plt.show()

#%% Data dictionary to use in models

data_dict = dict(
    obs=torch.tensor(data, device=device),
    # batch_size=10,
    m_mean=torch.tensor(m_guess, device=device),  # prior mean
    m_std=torch.tensor(40.0, device=device),  # prior std
    phase_min=torch.tensor(0.0, device=device),  # prior mean
    phase_max=torch.tensor(2 * np.pi, device=device),  # prior std
    phase_guess=torch.tensor(phase_guess, device=device),
    phase_mean=torch.tensor(0.0, device=device),
    phase_std=torch.tensor(10.0, device=device),  # prior std
    phase_concentration=torch.tensor(1e-4, device=device),
    device=device,
    t=torch.tensor(t, device=device),
    noise_std=torch.tensor(noise_std, device=device),
)

# Fit model
optimizer = torch.optim.Adam
optim_params = {
    "optimizer": optimizer,
    "optim_args": {"lr": 0.1},
    "step_size": 1000,
    "gamma": 0.8,
}
# this creates the equivalent dictionary as the one above
#optim_params = dict(
#    optimizer= optimizer,
#    optim_args= {"lr": 0.1},
#    step_size= 1000,
#    gamma= 0.8)
scheduler = pyro.optim.StepLR(optim_params)
# initialize optimization
pyro.clear_param_store()
pyro.set_rng_seed(1)
svi = SVI(
    models.waveguide,
    models.variational_waveguide,
    scheduler,
    loss=Trace_ELBO(),
)
# profiler = cProfile.Profile()
iter_loss = []
num_iterations = 10000
for j in tqdm(range(num_iterations)):
    # calculate the loss and take a gradient step
    # profiler.enable()
    # start_iter = time.time()
    loss = svi.step(data_dict)
    iter_loss.append(loss)
    scheduler.step()
    # elapsed = (time.time() - start_iter) / 60

params = dict()
for key in pyro.get_param_store().get_all_param_names():
    params[key] = pyro.param(key).cpu().detach().numpy()
print("m: ", params["m_mean"], params["m_std"])
print("phase: ", params["phase_mean"],params["phase_std"])
# %% Visualize reconstruction
y_reconstructed = np.sin(params["m_mean"] * t + params["phase_mean"])
plt.plot(t, y)
plt.plot(t, data)
plt.plot(t, y_reconstructed)
plt.legend(labels=["True", "Measured", "Reconstructed"])
plt.title("m_mean: "
    + str(np.round(params["m_mean"], decimals=2))
    + ", std: "
    + str(np.round(params["m_std"], decimals=2))
    + ",phase_mean: "
    + str(np.round(params["phase_mean"], decimals=2))
    # + ", phase_conc: "
    + ", std"
    + str(np.round(params["phase_std"], decimals=2))
)
plt.show()

# %%
