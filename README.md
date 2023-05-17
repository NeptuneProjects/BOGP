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
|   +-- swellex96
|   |   +-- data
|   |   |   +-- acoustics.yaml
|   |   |   +-- mfp.yaml
|   |   +-- optimization
|   |   |   +-- common.py
|   |   |   +-- configure.py
|   |   |   +-- run.py
```
The subdirectory `conf/swellex96/data` implements `yaml` files that are used with the [`Hydra`](https://hydra.cc) configuration management framework.
These configuration files control acoustic data preprocessing steps for later use in the workflow.

The subdirectory `conf/swellex96/optimization` implements Python modules which are used by the [`hydra-zen`](https://mit-ll-responsible-ai.github.io/hydra-zen/) configuration management framework.
These modules are used to configure the optimization workflow.

Both the `Hydra` and `hydra-zen` frameworks are implemented as I was interested in comparing the two.

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
bash ./scripts/process.sh
```

### High-resolution Matched Field Processing (MFP)
High-resolution MFP is performed to serve as a baseline comparison for other optimization methods.
To edit the MFP configuration, edit the following file:  
`conf/swellex96/data/mfp.yaml`

To run MFP, run the following from the `src` directory:
```bash
python3 ./data/swellex96/mfp.py
```

### Optimization
