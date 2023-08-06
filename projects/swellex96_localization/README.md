# Bayesian optimization with Gaussian process surrogate model for acoustic source localization

This repository contains the code used to perform acoustic source localization using Bayesian optimization with a Gaussian process surrogate model. This work was submitted to the Journal of the Acoustical Society of America:

> William Jenkins, Peter Gerstoft, and Yongsung Park. "Bayesian optimization with Gaussian process surrogate model for acoustic source localization," submitted to the Journal of the Acoustical Society of America, 31 May 2023.


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
|   +-- data
|   |   +-- acoustics.yaml
|   |   +-- mfp.yaml
|   +-- optimization
|   |   +-- common.py
|   |   +-- configure.py
|   |   +-- run.py
```
The subdirectory `conf/data` implements `yaml` files that are used with the [`Hydra`](https://hydra.cc) configuration management framework.
These configuration files control acoustic data preprocessing steps for later use in the workflow.

The subdirectory `conf/optimization` implements Python modules which are used by the [`hydra-zen`](https://mit-ll-responsible-ai.github.io/hydra-zen/) configuration management framework.
These modules are used to configure the optimization workflow.

Both the `Hydra` and `hydra-zen` frameworks are implemented as I was interested in comparing the two. I think I prefer the `hydra-zen` framework.

### Source Data Directories
Experimental data directories are organized as follows:
```
../../
+-- data
|   +-- swellex96_S5_VLA
|   |   +-- acoustic
|   |   |   +-- ambiguity_surfaces
|   |   |   |   +-- 148-166-201-235-283-338-388_200x100
|   |   |   |   |   +-- grid_parameters.pkl
|   |   |   |   |   +-- parameterization.json
|   |   |   |   |   +-- surface_000.npy
|   |   |   |   |   +-- ...
|   |   |   |   |   +-- surface_349.npy
|   |   |   +-- processed
|   |   |   |   +-- 49.0Hz
|   |   |   |   |   +-- covariance.npy
|   |   |   |   |   +-- data_49Hz.mat
|   |   |   |   |   +-- data.npy
|   |   |   |   |   +-- f_hist.npy
|   |   |   |   +-- ...
|   |   |   |   +-- 388.0Hz
|   |   |   |   +-- SBL
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
bash ./projects/swellex96_localization/scripts/process.sh
```

### Building the Environment Model `json` File
The environment model for SWellEx-96 is built by running the following script from the `src` directory:
```bash
bash ./projects/swellex96_localization/scripts/build_at_env.sh
```

### High-resolution Matched Field Processing (MFP)
High-resolution MFP is performed to serve as a baseline comparison for the other optimization methods.
To configure MFP, edit the following file:  
`conf/data/mfp.yaml`

To run MFP, run the following from the `src` directory:
```bash
python3 ./projects/swellex96_localization/data/mfp.py
```

### Optimization
There are two primary optimization configuration files.
- In `conf/optimization/run.py`, the individual optimization strategies are configured.
- In `conf/optimization/configure.py`, an entire optimization workflow is configured.

The file `conf/optimization/common.py` contains configurations common to individual optimization strategies and the broader experiment.

To generate a queue of optimization configurations that can be run sequentially or in parallel, run the following from the `src` directory:
```bash
bash ./projects/swellex96_localization/scripts/config.sh <serial name> <mode | experimental,simulation>
```

To run an optimization serial (a batch of configurations), run the following from the `src` directory:
```bash
bash ./projects/swellex96_localization/scripts/run.sh <path to queue> <num jobs>
```
The queue can be run in parallel by specifying `<num jobs>`.

### Aggregate Optimization Results
To aggregate the results of an optimization serial, run the following from the `src` directory:
```bash
bash ./projects/swellex96_localization/scripts/agg.sh <path to queue>
```
Edit `agg.sh` directly to specify which serial to aggregate.
The output of this script results in two `.csv` files:
- `aggregated_results.csv` contains the results of each step of all optimization configurations concatenated into one `csv` file.
- `best_results.csv` contains only the optimal results from each optimization configuration.

### Collate Results
For use in producing figures, a final, collated `.csv` is created by collating data and metadata from various sources in the epxerimental data.
To collate results, run the following from the `src` directory:
```bash
bash ./projects/swellex96_localization/scripts/collate.sh
```
Edit `collate.sh` directly to specify which serial to aggregate.

### Plotting Results
To plot results, run the following from the `src` directory:
```bash
python3 ./projects/swellex96_localization/plotting/figures.py --figures=<1,2,5,...>
```
The `--figures` argument is a comma-separated list of figures to plot.
