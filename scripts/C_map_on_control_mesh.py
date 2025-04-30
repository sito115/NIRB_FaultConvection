from ast import List
import sys
import numpy as np
import pyvista as pv
from pathlib import Path
import pandas as pd
from helpers import safe_parse_quantity, delete_pyvista_fields, map_on_control_mesh
import vtk
from tqdm import tqdm
from typing import Tuple
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ComsolClasses.comsol_classes import COMSOL_VTU
from ComsolClasses.helper import calculate_normal


def create_control_mesh(bounds: pv.BoundsLike,
                        spacing: Tuple[int | float, int | float, int | float]) -> vtk.vtkImageData:
    """Generates a structured grid withtin "bounds" and spacing in x, y, z directions specified in "spacing".

    Args:
        bounds (pv.BoundsLike): _description_
        spacing (Tuple[int  |  float, int  |  float, int  |  float]): _description_

    Returns:
        vtk.vtkImageData: _description_
    """
    dx, dy, dz = spacing  # You can increase this for more resolution
    # Compute the number of points needed (+ 1 as spacing is an interval)
    nx = int((bounds[1] - bounds[0]) / dx) + 1
    ny = int((bounds[3] - bounds[2]) / dy) + 1
    nz = int((bounds[5] - bounds[4]) / dz) + 1

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetOrigin(bounds[0], bounds[2], bounds[4])
    image.SetSpacing(dx, dy, dz)
    return image


def main():   
    ROOT = Path(__file__).parents[1]
    PARAMETER_SPACE = "03"
    DATA_TYPE = "Training"
    FIELDS_TO_EXPORT : List[str] = ["Temperature"]
    TIME_STEPS_TO_EXPORT : List[str | int] = [-1]  

    data_folder = Path(ROOT / "Snapshots" / PARAMETER_SPACE /  "Training_Original") # data_type) #"Truncated") # data_type)    
    assert data_folder.exists(), f"Data folder {data_folder} does not exist."
    export_folder =  data_folder.parent / DATA_TYPE
    assert export_folder.exists(), f"Export folder {export_folder} does not exist."
    assert str(data_folder.absolute()) != str(export_folder.absolute()), "Import and Export from same folder not allowed"

    vtu_files = sorted([path for path in data_folder.iterdir() if path.suffix == ".vtu"])
    
    bounds = COMSOL_VTU(vtu_files[0]).mesh.bounds
    control_mesh = create_control_mesh(bounds)


if __name__ == "__main__":
    main()