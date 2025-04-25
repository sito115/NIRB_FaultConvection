import numpy as np
from pathlib import Path
import pandas as pd
import pint
import pint_pandas

def min_max_scaler(data: np.ndarray) -> np.ndarray:
    """Min-max scaler to scale the data between 0 and 1.

    Args:
        data (np.ndarray): Data to be scaled.

    Returns:
        np.ndarray: Scaled data.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def inverse_min_max_scaler(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Inverse min-max scaler to scale the data back to its original range.

    Args:
        data (np.ndarray): Data to be scaled back.
        min_val (float): Minimum value of the original data.
        max_val (float): Maximum value of the original data.

    Returns:
        np.ndarray: Scaled back data.
    """
    return data * (max_val - min_val) + min_val

def safe_parse_quantity(s, ureg: pint.UnitRegistry = pint.UnitRegistry()):
    try:
        return ureg(s)
    except Exception:
        return np.nan


def load_pint_data(path: Path, **kwargs) -> pd.DataFrame:
    header = kwargs.pop("header", [0, 1])
    level = kwargs.pop("level", -1)
    training_param = pd.read_csv(path, header=header)
    training_param =  training_param.pint.quantify(level = level)
    is_numpy = kwargs.pop("is_numpy", False)
    if is_numpy:
        return training_param.pint.dequantify().to_numpy()
    return training_param
        
def standardize(array: np.ndarray, mean:np.ndarray, var:np.ndarray) -> np.ndarray:
    return (array - mean)/np.sqrt(var)

    
if __name__ == "__main__":
    param_path = Path("Surrogate_NIRB/test_samples.csv")
    load_pint_data(param_path)