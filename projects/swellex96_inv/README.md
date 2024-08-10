# Bayesian optimization for geoacoustic inversion

This repository contains the code used to perform geoacoustic inversion using Bayesian optimization with a Gaussian process surrogate model. This work has been submitted to the Journal of the Acoustical Society of America and is currently in review:

> W. F. Jenkins, P. Gerstoft, and Y. Park, “Geoacoustic inversion using Bayesian optimization with a Gaussian process surrogate model,” J. Acoust. Soc. Am., vol. 156, no. 2, pp. 812–822, Aug. 2024, doi: [10.1121/10.0028177](https://doi.org/10.1121/10.0028177).

## Installation

In your desired target directory, run the following command:
```bash
git clone git@github.com:NeptuneProjects/BOGP.git
```

Once cloned, build the Conda environment.
This may take a few minutes.
Two dependencies, [TritonOA](https://github.com/NeptuneProjects/TritonOA) and [OAOptimization](https://github.com/NeptuneProjects/OAOptimization), are automatically installed via `pip` from their respective GitHub repositories.
```bash
conda env create -f gp_dev.yml
```

Activate the environment:
```bash
conda activate gp310
```

## Usage

This workflow has been broken into multiple steps with corresponding scripts.
In general, configurations are split out from the scripts themselves.
This allows for easier modification of the workflow without having to modify the scripts.

### Configuration
The configuration scheme is organized as follows:
```
.
+-- conf
|   +-- acoustics.yaml
|   +-- mfp.yaml
+-- data
|   +-- bo
|   |   +-- common.py
```
The directory `conf` implements `yaml` files that are used with the [`Hydra`](https://hydra.cc) configuration management framework.
These configuration files control acoustic data preprocessing steps for later use in the workflow.

The subdirectory `data/bo` contains Python modules used for Bayesian optimization.
The `common.py` module contains configurations common to individual optimization strategies and the broader experiment.

### Source Data Directories
Experimental data directories are organized as follows:
```
../../
+-- data
|   +-- swellex96_S5_VLA_inv
|   |   +-- acoustic
|   |   |   +-- processed
|   |   |   |   +-- 148.0Hz
|   |   |   |   |   +-- covariance.npy
|   |   |   |   |   +-- data_201Hz.mat
|   |   |   |   |   +-- data.npy
|   |   |   |   |   +-- f_hist.npy
|   |   |   |   +-- ...
|   |   |   |   +-- 388.0Hz
|   |   |   |   +-- merged.npz
|   |   |   +-- raw
|   |   |   |   +-- mat
|   |   |   |   +-- npy
|   |   |   |   +-- sio
|   |   |   |   |   +-- J131
|   |   |   |   |   +-- J132
|   |   +-- ctd
|   |   |   +-- i9601.prn
|   |   |   +-- ...
|   |   +-- env_models
|   |   |   +-- swellex96.json
|   |   +-- geoacoustic
|   |   +-- gps
|   |   |   +-- gps_range.csv
```

### Data Preparation
Acoustic source localization is performed using experimental data from the [SWellEx-96 experiment](http://swellex96.ucsd.edu).
The following steps are taken in pre-processing:
1. Convert `.sio` data files to `.npy` files.
2. Merge `.npy` files into a single `.npy` file according to a date/time range.
3. Process the merged `.npy` file into files containing complex pressures by a defined time step and frequency.
4. Compute the covariance matrix for each time step and frequency.

To configure the data preparation steps, make changes to the following file:  
`conf/swellex96/data/acoustics.yaml`  
The first lines of `acoustics.yaml` control which pre-processing steps to take. Comment out the steps that you wish to skip.
```yaml
run:
  - convert  # Step 1
  - merge  # Step 2
  - process  # Step 3
  - compute_covariance  # Step 4
```

To perform pre-processing, run the following script from the `src` directory:
```bash
bash ./projects/swellex96_inv/scripts/process.sh
```

### Building the Environment Model `json` File
The environment model for SWellEx-96 is built by running the following script from the `src` directory:
```bash
bash ./projects/swellex96_inv/scripts/build_at_env.sh
```

### Optimization
Optimization is run using the following script:
```bash
bash ./projects/swellex96_inv/scripts/run.sh
```

To adjust the optimization configuration, make changes to `run.sh` directly.
The syntax for individual lines in the script can be queried by running:
```bash
python ./projects/swellex96_inv/data/bo/run.py --help
```
Nine optimization strategies are available:
- `ucb`: Bayesian optimization with upper confidence bound acquisition function.
- `pi`: Bayesian optimization with probability of improvement acquisition function.
- `ei`: Bayesian optimization with expected improvement acquisition function.
- `logei`: Bayesian optimization with logarithmic expected improvement acquisition function.
- `gibbon`: A subspace Bayesian optimization strategy.
- `baxus`: A subspace Bayesian optimization strategy.
- `grid`: A gridded direct-search optimization strategy.
- `random`: A random direct-search optimization strategy.
- `sobol`: A Sobol sequence direct-search optimization strategy.

Optimization results are saved to:
```
../../
+-- data
|   +-- swellex96_S5_VLA_inv
|   |   +-- outputs
|   |   |   +-- runs
|   |   |   |   +-- <optimization strategy>
```

### Plotting Results
To plot results, run the following from the `src` directory:
```bash
python3 ./projects/swellex96_inv/visualization/figures.py
```
