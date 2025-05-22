# Convection along Fault Planes

The benchmark model from 
- Magri, F., Cacace, M., Fischer, T., Kolditz, O., Wang, W., & Watanabe, N. (2017). Thermal convection of viscous fluids in a faulted system: 3D benchmark for numerical codes. Energy Procedia, 125, 310–317. https://doi.org/10.1016/j.egypro.2017.08.204

is extended to investigate the factors influencing convection-induced thermal anomalies along fault planes. The benchmark model looks like this:
![Convection Model](docs/Temperature_Benchmark_3D_Fault.png)


This project aims to create a surrogate model for a 3D cube containing a fault. The simulations are exported from COMSOL and stored in the `data` folder. Each subdirectory is labeled with a number (e.g., "01"), corresponding to a specific parameter space, which is declared in the respective `training_samples.csv`.

Each script/notebook contains an alphabetical prefix that denotes its chronological order. In general, `*.py` files are used for calculations and computations, while `*.ipynb` files are used for quality checks and plotting.

````shell
├── data
│   ├── 01 # Parameter Space 01 
│   ├── 02 # Parameter Space 02 
│   ├── 03 # Parameter Space 03 
│   └── ..
├── docs
├── notebooks
│   ├── A_plot_samples.ipynb
│   ├── C_entropy_vs_n_cells.ipynb
│   ├── C_plot_scaled_vs_unscaled_data.ipynb
│   ├── C_quality_check_plots.ipynb
│   ├── D_quality_check.ipynb
│   ├── E_analyse_sweep.ipynb
│   ├── E_online_stage.ipynb
│   ├── F_sensitivity_analysis.ipynbb
├── scr
│   ├── comsol_module
│   ├── config.ini
│   ├── MPh
│   ├── offline_stage
│   ├── pod
│   ├── SALib
│   ├── sampling
│   └── utils
├── scripts
│   ├── A_sampling.py
│   ├── B_compute_snapshots.py
│   ├── C_map_on_control_mesh.py
│   ├── C_merge_plot_snapshots.py
│   ├── D_pod.py
│   ├── E_coefficients_model.py
│   ├── E_sweep.py
│   └── Z_write_config.pyy
````


## Overview

### A - Sampling
This script defines the parameter space, specifying which parameters can vary in the surrogate model and their respective ranges. The samples are exported as CSV files using the `pint` package, an elegant tool for unit handling.

### B - Loop (Generating Snapshots)
This script is intended for use on computers where the COMSOL solver is available. It loads the parameters from the CSV file, inserts them into the COMSOL model, and then runs the simulation. The results are saved as `.vtu` files.

### C - Quality Check
This step extracts data from the `.vtu` files and merges it into `.npy` files. Additionally, plots and movies are generated to visualize and verify the simulations.

### D - POD 
This step involves applying Proper Orthogonal Decomposition (POD) to the simulation data.
Suffixes for normalization (mandatory):
- `min_max` - Min max normalization (with MinMaxScaler)
- `mean` - Mean normalization (with MeanScaler)
- ... `init` - First time step is substracted
- ... `init_grad` - Analytically calculcated temperature gradient is substracted from solution.

For example, the 

### E - Coefficient Model
In this step, a machine learning model is created using the `Lightning` library to estimate the coefficients for each basis function.
