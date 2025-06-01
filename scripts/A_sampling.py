"""
This script generates parameter samples for training and testing snapshots.
The parameter space is defined by adding multiple instances of the "Parameter" class.
Training samples are generated using hypercube sampling, while test samples use random sampling.
The resulting samples are saved to "training_samples.csv" and "test_samples.csv".
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pyDOE import lhs
from typing import List
import random
import pint_pandas
import pint # noqa: F401
import sys
sys.path.append(str(Path(__file__).parents[1]))
from src.sampling import Parameter



def main():
    ### SETUP
    ROOT = Path(__file__).parent
    N_TRAINING = 0
    N_TEST = 20
    pint_pandas.PintType.ureg.formatter.default_format = "#D~"
    random.seed(42)

    
    ### DEFINE PARAMETERS
    parameters: List[Parameter] = []
    # parameters.append(Parameter("fault_k_trans", [1e-18, 1e-12], unit="m^2", is_log=True))
    # parameters.append(Parameter("fault_k_long", [1e-18,1e-12], unit="m^2", is_log=True))
    # parameters.append(Parameter("host_k", [1e-18, 1e-15],  unit="m^2", is_log=True))
    # parameters.append(Parameter("T_h", [195, 220], unit = "degC", is_log=False))
    parameters.append(Parameter("dip", [50, 90], unit = "deg", is_log=False))

    units_dict = {param.name: f'pint[{param.unit}]' for param in parameters}
    
    ### GENERATE TRAINING SAMPLES (LHS)
    if N_TRAINING > 0:
        lhd = lhs(len(parameters), samples=N_TRAINING, criterion="center")
        print(lhd)

        lhd_scaled = np.array(
            [param.scale_from_lhs(lhd[:, i]) for i, param in enumerate(parameters)]
        ).T
        print(lhd_scaled)

        df_traing = pd.DataFrame(lhd_scaled, columns=[param.name for param in parameters])
        df_traing = df_traing.astype(units_dict)
        
        df_traing = df_traing.pint.to_base_units().pint.dequantify()
        df_traing.to_csv(ROOT / "training_samples.csv", index=False)
        print(df_traing)

    ### GENERATE TEST SAMPLES (RANDOM)
    if N_TEST > 0:
        random_samples = np.random.rand(N_TEST, len(parameters))
        random_samples_scaled = np.array(
            [
                param.scale_from_lhs(random_samples[:, i])
                for i, param in enumerate(parameters)
            ]
        ).T
        df_test = pd.DataFrame(
            random_samples_scaled, columns=[param.name for param in parameters]
        )
        df_test = df_test.astype(units_dict)
        df_test = df_test.pint.to_base_units().pint.dequantify()

        df_test.to_csv(ROOT / "test_samples.csv", index=False)
        print(df_test)

if __name__ == "__main__":
    main()
