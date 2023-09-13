# Bayesian optimization with Gaussian process surrogate model for geoacoustic inversion and parameter estimation

[![DOI](https://zenodo.org/badge/564585214.svg)](https://zenodo.org/badge/latestdoi/564585214)

This repository contains code used to perform acoustic parameter estimation using Bayesian optimization with a Gaussian process surrogate model. The following papers use this code:

> William Jenkins, Peter Gerstoft, and Yongsung Park, “Bayesian optimization with Gaussian process surrogate model for source localization,” J Acoust. Soc. Am., vol. 154, no. 3, pp. 1459–1470, Sep. 2023, doi: [10.1121/10.0020839](https://doi.org/10.1121/10.0020839).

> William Jenkins and Peter Gerstoft. "Bayesian optimization with Gaussian processes for robust localization," submitted to IEEE Int. Conf. Acoust., Speech, Signal Process., Sep. 2023.

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

This workflow applies to multiple projects and data sets. Specific instructions for running the workflow on a particular data set are provided in the corresponding `README.md` files:

| Application | Data | Instructions |
----------|------|--------------|
| Acoustic source localization | SWellEx-96 | [README.md](projects/swellex96_localization/README.md)
| Source localization robust to array tilt | SWellEx-96 | [README.md](projects/swellex96_loc_tilt/README.md)