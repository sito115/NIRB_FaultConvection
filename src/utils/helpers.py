import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from pathlib import Path
import pint
from src.comsol_module.src.comsol_module import calculate_S_therm, caluclate_entropy_gen_number_isotherm
import pyvista as pv
from typing import Tuple, Literal, Any
import logging
import pandas as pd

ureg = pint.get_application_registry()

def mse(predictions :np.ndarray , targets: np.ndarray) -> np.floating[Any]:
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
    return float(toReturnQ2)

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
    return float(toShowR2)


def calculate_thermal_entropy_generation(ref_mesh : pv.DataSet,
                                         data : np.ndarray,
                                         lambda_therm : pint.Quantity,
                                         t0: pint.Quantity,
                                         delta_T: pint.Quantity,
                                         bc : Literal['isotherm', 'isoflux'] = 'isotherm') -> Tuple[pint.Quantity, pint.Quantity]:  
    """_summary_

    Args:
        ref_mesh (pv.DataSet): _description_
        data (np.ndarray): must match n_points of ref_mesh
        lambda_therm (pint.Quantity): _description_
        t0 (pint.Quantity): _description_
        delta_T (pint.Quantity): _description_
        ureg (pint.UnitRegistry): _description_

    Returns:
        Tuple[pint.Quantity]: (s0_total [W/(K)], entropy_number [-]), Volumetric entropy generation [W/(K*m^3)]

    """      
    ref_mesh.clear_data()
    ref_mesh.point_data["temp_field"] = data
    try:
        ref_mesh = ref_mesh.clean(progress_bar = False)
    except AttributeError as e:
        # logging.warning(e)
        pass
    
    temp_grad = ref_mesh.compute_derivative("temp_field", preference = "point").point_data["gradient"] * ureg.kelvin / ureg.meter
    s0 = calculate_S_therm(lambda_therm,
                           t0,
                           temp_grad)
    
    ref_mesh.point_data["temp_field"] = s0.magnitude 
    integrated = ref_mesh.integrate_data()
    s0_total : pint.Quantity = integrated.point_data['temp_field'][0] * ureg.watt / ureg.kelvin
    
    match bc:
        case 'isotherm':
            # L = (ref_mesh.bounds.z_max - ref_mesh.bounds.z_min) * ureg.meter
            L = (ref_mesh.bounds[-1] - ref_mesh.bounds[-2]) * ureg.meter

            entropy_number : pint.Quantity = caluclate_entropy_gen_number_isotherm(s_total=s0_total,
                                                                L = L, lambda_m=lambda_therm,
                                                                T_0 = t0,
                                                                V = ref_mesh.volume * ureg.meter**3,
                                                                delta_T=delta_T)
        case 'isoflux':
            raise NotImplementedError()
        
    assert entropy_number.check(['dimensionless']) # check for correct unit
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
        
        
def find_snapshot_path(projection: Literal['Mapped', 'Original'],
                       suffix : str,
                       field_name: Literal['Temperature', 'Entropy'],
                       root_data_folder : Path,
                       control_mesh_suffix : str,
                       data_type: Literal['Training', 'Test']) -> np.ndarray:
    
    match projection:
        case"Mapped":
            export_root = root_data_folder / f"{data_type}{projection}" / control_mesh_suffix / "Exports"
        case "Original":
            export_root = root_data_folder / f"{data_type}{projection}" 
        
    assert export_root.exists(), f"Export root {export_root} does not exist."
    
    match field_name:
        case "Temperature":
            if 'init' in suffix.lower() and 'grad' in suffix.lower():
                logging.info(f"Entered {data_type}_Temperature_minus_tgrad.npy")
                snapshots_npy      = np.load(export_root / f"{data_type}_Temperature_minus_tgrad.npy")
            elif 'init' in suffix.lower():
                snapshots_npy      = np.load(export_root / f"{data_type}_Temperature.npy")
                snapshots_npy  = snapshots_npy -  snapshots_npy[:, 0, :] # last time step
            else:
                logging.info(f"Entered {data_type}_Temperature.npy")
                snapshots_npy      = np.load(export_root / f"{data_type}_Temperature.npy")
        case "Entropy":
            snapshots_npy      = np.load(export_root / f"{data_type}_entropy_gen_per_vol_thermal.npy")

    return snapshots_npy


def find_basis_functions(projection: Literal['Mapped', 'Original'],
                         suffix : str,
                         accuracy: float,
                         field_name: Literal['Temperature', 'Entropy'],
                         root_data_folder : Path,
                         control_mesh_suffix : str) -> np.ndarray:
    match projection:
        case "Mapped":
            basis_func_path = root_data_folder / "TrainingMapped" / control_mesh_suffix / f"BasisFunctions{field_name}" / f"basis_fts_matrix_{accuracy:.1e}{suffix}.npy"
        case "Original":
            basis_func_path = root_data_folder / "TrainingOriginal" / f"BasisFunctions{field_name}" / f"basis_fts_matrix_{accuracy:.1e}{suffix}.npy"
    
    assert basis_func_path.exists()
    logging.info(f"{basis_func_path=}")
    return np.load(basis_func_path)
    