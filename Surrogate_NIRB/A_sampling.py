from pathlib import Path
import numpy as np
import pandas as pd
from pyDOE import lhs
from dataclasses import dataclass
import enum
from typing import List
from scipy.stats.distributions import norm
import random
import pint_pandas
import pint
from pint.delegates.formatter._format_helpers import formatter

ureg = pint.UnitRegistry()
# Create a custom Unit class that overrides the default string representation
@pint.register_unit_format("COMSOL")
def format_unit_simple(unit, registry, **options):
    return " * ".join(f"{u} ** {p}" for u, p in unit.items())


class Distribution(enum.Enum):
    uniform = "uniform"
    normal = "normal"
    lognormal = "lognormal"

@dataclass
class Parameter:
    name: str
    param_range: list
    unit : str = None
    is_log: bool = False

    def scale_from_lhs(self, lhs_samples):
        assert self.param_range[0] < self.param_range[1], (
            f"{self.name}: Range should be in ascending order"
        )

        if self.is_log:
            param_range = np.log10(self.param_range)
            low, high = param_range[0], param_range[1]
        else:
            low, high = self.param_range[0], self.param_range[1]

        samples = (high - low) * lhs_samples + low

        if self.is_log:
            samples = 10**samples

        return samples

    def mean(self):
        if self.is_log:
            return np.mean(np.log10(self.param_range))
        else:
            return np.mean(self.param_range)

def main():
    ### SETUP
    ROOT = Path(__file__).parent
    N_TRAINING = 100
    N_TEST = 20
    pint_pandas.PintType.ureg.formatter.default_format = "#D~"
    ### DEFINE PARAMETERS
    parameters: List[Parameter] = []
    # parameters.append(Parameter("fault_k_trans", [1e-18, 1e-12], unit="m^2", is_log=True))
    # parameters.append(Parameter("fault_k_long", [1e-18,1e-12], unit="m^2", is_log=True))
    # parameters.append(Parameter("host_k", [1e-18, 1e-15],  unit="m^2", is_log=True))
    # parameters.append(Parameter("T_h", [130, 220], unit = "degC", is_log=False))
    parameters.append(Parameter("dip", [50, 90], unit = "deg", is_log=False))

    units_dict = {param.name: f'pint[{param.unit}]' for param in parameters}
    
    ### TRAINING SAMPLES
    lhd = lhs(len(parameters), samples=N_TRAINING, criterion="center")
    print(lhd)

    lhd_scaled = np.array(
        [param.scale_from_lhs(lhd[:, i]) for i, param in enumerate(parameters)]
    ).T
    print(lhd_scaled)
    means = np.mean(lhd_scaled, axis=0)

    for i, param in enumerate(parameters):
        print(f"{param.name}: {param.mean()}")
        print(f"{param.name}: {means[i]}")
        print(f"{param.name}: {np.mean(np.log10(lhd_scaled[:, i]))}")

    

    df_traing = pd.DataFrame(lhd_scaled, columns=[param.name for param in parameters])
    df_traing = df_traing.astype(units_dict)
    

    df_traing = df_traing.pint.to_base_units().pint.dequantify()
    df_traing.to_csv(ROOT / "training_samples.csv", index=False)
    print(df_traing)

    ### TEST SAMPLES
    random.seed(42)
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


if __name__ == "__main__":
    main()
