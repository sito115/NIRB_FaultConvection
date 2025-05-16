import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from pathlib import Path
import pint
from src.comsol_module.src.comsol_module import calculate_S_therm
import pyvista as pv
from typing import Tuple
import logging
import pandas as pd

def mse(predictions :np.ndarray , targets: np.ndarray) -> float:
    """Compute the Mean Squared Error (MSE) between predictions and targets.

    Args:
        predictions (np.ndarray): _description_
        targets (np.ndarray): _description_

    Returns:
        float: _description_
    """    
    return np.mean((predictions - targets)**2)    
    
    
def plot_data(data: np.ndarray, **kwargs):
    """_summary_

    Args:
        data (np.ndarray): _description_
        title (str): _description_
        export_path (Path): _description_
    """    
    fig, ax = plt.subplots(figsize = (12, 4))
    n_points = np.arange(data.shape[1])
    for array in data:
        ax.step(n_points,
                array,
                alpha = 0.5,
                linewidth = 0.5)
    title_string = kwargs.pop("title", "")
    export_path : Path = kwargs.pop("export_path", None)
    ax.set_xlabel("Point ID")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(title_string)
    ax.grid()
    if export_path is not None:
        assert export_path.parent.exists()
        fig.savefig(export_path)
    plt.close("all")
        

def Q2_metric(test_snapshots : np.ndarray, test_predictions: np.ndarray) -> float:
    """Test

    Args:
        test_snapshots (np.ndarray): _description_
        test_predictions (np.ndarray): _description_

    Returns:
        float: _description_
    """
    Q2 = mean_squared_error(test_snapshots, test_predictions, multioutput='raw_values')  
    # TODO: why mutliplied with - 1 in source code? 
    toReturnQ2 = np.average(Q2) #*(-1)  
    return toReturnQ2

def R2_metric(training_snapshots  : np.ndarray, training_predictions  : np.ndarray) -> float:
    """ Training

    Args:
        training_snapshots (np.ndarray): _description_
        training_predictions (np.ndarray): _description_

    Returns:
        float: _description_
    """
    R2 = mean_squared_error(training_snapshots, training_predictions, multioutput='raw_values')  
    toShowR2 = np.average(R2)
    return toShowR2


def calculate_thermal_entropy_generation(ref_mesh : pv.DataSet,
                                         data : np.ndarray,
                                         lambda_therm : pint.Quantity,
                                         t0: pint.Quantity,
                                         delta_T: pint.Quantity,
                                         ureg : pint.UnitRegistry) -> Tuple[pint.Quantity]:  
    """_summary_

    Args:
        ref_mesh (pv.DataSet): _description_
        data (np.ndarray): must match n_points of ref_mesh
        lambda_therm (pint.Quantity): _description_
        t0 (pint.Quantity): _description_
        delta_T (pint.Quantity): _description_
        ureg (pint.UnitRegistry): _description_

    Returns:
        Tuple[pint.Quantity]: (s0_total [W/(K * m^3)], entropy_number [-])

    """      
    ref_mesh.point_data["temp_field"] = data
    cell_mesh = ref_mesh.point_data_to_cell_data()
    temp_grad = cell_mesh.compute_derivative("temp_field").cell_data["gradient"] * ureg.kelvin / ureg.meter
    s0 = calculate_S_therm(lambda_therm,
                           t0,
                           temp_grad)
    s0_total = np.sum(s0 * ref_mesh.compute_cell_sizes()["Volume"] * ureg.meter**3)
    L = (ref_mesh.bounds.z_max - ref_mesh.bounds.z_min) * ureg.meter
    s0_characteristic = (lambda_therm * delta_T**2) / (L**2 * t0**2)
    entropy_number = s0_total / s0_characteristic / (ref_mesh.volume * ureg.meter**3)  
    assert entropy_number.check(['dimensionless'])
    return s0_total, entropy_number


def setup_logger(is_console: bool, log_file: Path = None, level = logging.DEBUG):
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler to log to a file
    # Stream handler to log to the console
    if is_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        
def format_comsol_unit(unit : str) -> str:
    """Make unit in format for COMSOL "value[unit]" and replace ** exponent with ^.

    Args:
        unit (str): _description_

    Returns:
        str: formatted unit
    """    
    return "[" + unit.replace("**", "^") + "]"


def check_range_is_valid(allowed_range: pd.DataFrame, user_range: pd.DataFrame):
    for col in user_range.columns:
        min2 = float(allowed_range[col].min())
        max2 = float(allowed_range[col].max())
        min1 = float(user_range[col].min())
        max1 = float(user_range[col].max())
        
        assert min2 <= min1 <= max1 <= max2, f"Column {col} has out-of-bound values: {min1}, {max1} not in [{min2}, {max2}]"