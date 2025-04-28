import numpy as np
from pathlib import Path
import pandas as pd
import pint
from typing import List
import pyvista as pv
from lightning.pytorch.callbacks import EarlyStopping
import vtk
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ComsolClasses.comsol_classes import COMSOL_VTU

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


def load_pint_data(path: Path, **kwargs) -> pd.DataFrame:
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
    is_numpy = kwargs.pop("is_numpy", False)
    if is_numpy:
        return training_param.pint.dequantify().to_numpy()
    return training_param
        
def standardize(array: np.ndarray, mean:np.ndarray, var:np.ndarray) -> np.ndarray:
    return (array - mean)/np.sqrt(var)

    
    
    
class MyEarlyStopping(EarlyStopping):
    
    def __init__(self, termination_threshold=5.0, min_epochs=5000, **kwargs):
        super().__init__(**kwargs)
        self.termination_threshold = termination_threshold
        self.min_epochs = min_epochs
    
    def on_train_end(self, trainer, pl_module):
        # Check only at the end of training
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            print(f"Warning: {self.monitor} not found in metrics.")
            return
        
        current_epoch = trainer.current_epoch
        
        if current_epoch >= self.min_epochs and current > self.termination_threshold:
            print(f"Early Stopping: train_loss = {current:.4e} after {current_epoch} epochs (> {self.termination_threshold})")
            trainer.should_stop = True
        else:
            print(f"Training completed normally. train_loss = {current:.4e} after {current_epoch} epochs.")
    

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
        
    
if __name__ == "__main__":
    param_path = Path("Surrogate_NIRB/test_samples.csv")
    load_pint_data(param_path)