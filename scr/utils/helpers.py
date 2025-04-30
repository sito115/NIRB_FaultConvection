import numpy as np
from pathlib import Path
import pandas as pd
import pint_pandas  # noqa: F401
import pint
from typing import List
import pyvista as pv
import vtk
from sklearn.metrics import mean_squared_error
from scr.comsol_module.comsol_classes import COMSOL_VTU

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
    """Convert string quantities in Dataframe to pint quantities.

    Args:
        s (_type_): String quantity, e.g. "0.2 m*m"
        ureg (pint.UnitRegistry, optional): _description_. Defaults to pint.UnitRegistry().

    Returns:
        _type_: pint quantity
    """    
    try:
        return ureg(s)
    except Exception:
        return np.nan


def load_pint_data(path: Path, is_numpy = False, **kwargs) -> pd.DataFrame:
    """Load csv that has parameter names in the first row and units in the second row.

    Args:
        path (Path): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    header = kwargs.pop("header", [0, 1])
    level = kwargs.pop("level", -1)
    training_param = pd.read_csv(path, header=header)
    training_param =  training_param.pint.quantify(level = level)
    if is_numpy:
        return training_param.pint.dequantify().to_numpy()
    return training_param
        
def standardize(array: np.ndarray, mean:np.ndarray, var:np.ndarray) -> np.ndarray:
    """This function subtracts the mean and divides by the square root of the variance
    for each element, effectively transforming the input to have zero mean and unit variance.

    Args:
        array (np.ndarray): _description_
        mean (np.ndarray): _description_
        var (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """    
    
    return (array - mean)/np.sqrt(var)


def mse(predictions :np.ndarray , targets: np.ndarray) -> float:
    """Compute the Mean Squared Error (MSE) between predictions and targets.

    Args:
        predictions (np.ndarray): _description_
        targets (np.ndarray): _description_

    Returns:
        float: _description_
    """    
    return np.mean((predictions - targets)**2)    
    
    
def delete_pyvista_fields(comsol_data : COMSOL_VTU,
                     fields_2_keep : List[str] = "Temperature") -> COMSOL_VTU:
    """Deletes all fields in COMSOL_VTU.mesh except fields_2_keep.

    Args:
        comsol_data (COMSOL_VTU): _description_
        field_2_keep (List[str], optional): _description_. Defaults to "Temperature".

    Returns:
        COMSOL_VTU: _description_
    """
    fields_2_delete = comsol_data.exported_fields.copy()
    for field_2_keep in fields_2_keep:
        fields_2_delete.remove(field_2_keep)
    for idx, field in enumerate(fields_2_delete):
        comsol_data.delete_field(field)
    return comsol_data


def map_on_control_mesh(comsol_data : pv.PolyData,
                        control_mesh: vtk.vtkImageData) -> pv.ImageData:
    """Map 

    Args:
        comsol_data (pv.PolyData): _description_
        control_mesh (vtk.vtkImageData): _description_

    Returns:
        vtk.vtkImageData: _description_
    """    
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(control_mesh)        # The grid where you want data
    probe.SetSourceData(comsol_data)    # The mesh with the data to interpolate
    probe.Update()
    interpolated = probe.GetOutput()
    return pv.wrap(interpolated)
        

def Q2_metric(test_snapshots : np.ndarray, test_predictions: np.ndarray) -> float:
    """Test

    Args:
        test_snapshots (np.ndarray): _description_
        test_predictions (np.ndarray): _description_

    Returns:
        float: _description_
    """
    Q2 = mean_squared_error(test_snapshots, test_predictions, multioutput='raw_values')  
    toReturnQ2 = np.average(Q2)*(-1)  # TODO: why mutliplied with - 1 in source code? 
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


ureg = pint.UnitRegistry()
preferred_units = {
    ureg.Quantity(1, 'degree').dimensionality: 'degree',
    ureg.Quantity(1, 'degC').dimensionality: 'degC',
    ureg.Quantity(1, 'bar').dimensionality: 'bar',
}

def format_quantity(q: pint.Quantity) -> str:
    """Display a pint.Quanitity as string in "value unit" format.
    Additionally, preferred units are inserted for temperature, angles.

    Args:
        q (pint.Quantity): _description_

    Returns:
        str: _description_
    """    
    unit = preferred_units.get(q.dimensionality, q.units)
    try:
        q = q.to(unit)
    except pint.errors.UndefinedUnitError:
        pass  # Skip if conversion fails
    return f"{q.magnitude:.2e} {q.units:~P}"




if __name__ == "__main__":
    pass