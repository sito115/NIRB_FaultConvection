import numpy as np
import pyvista as pv
from typing import Tuple, List
import vtk
from scr.comsol_module.comsol_classes import COMSOL_VTU

def create_control_mesh(bounds: pv.BoundsTuple,
                        spacing: Tuple[int | float, int | float, int | float]) -> pv.ImageData:
    """Generates a structured grid withtin "bounds" and spacing in x, y, z directions specified in "spacing".

    Args:
        bounds (pv.BoundsLike): _description_
        spacing (Tuple[int  |  float, int  |  float, int  |  float]): _description_

    Returns:
        vtk.vtkImageData: _description_
    """
    dx, dy, dz = spacing  # You can increase this for more resolution
    # Compute the number of points needed (+ 1 as spacing is an interval)

    # Compute the number of points needed (+ 1 as spacing is an interval)
    nx = int(np.abs((bounds[1] - bounds[0])) / dx) + 1
    ny = int(np.abs((bounds[3] - bounds[2])) / dy) + 1
    nz = int(np.abs((bounds[5] - bounds[4])) / dz) + 1
    
    image = pv.ImageData() 
    image.origin = (bounds[0], bounds[2], bounds[4])
    image.dimensions = np.array((nx, ny, nz))
    image.spacing = (dx, dy, dz)
    
    return image


def delete_comsol_fields(comsol_data : COMSOL_VTU,
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
    """Map on control mesh with a vtkProbeFilter
    https://vtk.org/doc/nightly/html/classvtkProbeFilter.html
    https://public.kitware.com/Wiki/Demystifying_the_vtkProbeFilter

    VtkProbeFilter
    For structured datasets (vtkImageData, vtkStructuredGrid), VTK uses trilinear interpolation based on the grid spacing and cell values.
    For unstructured grids, interpolation is done using barycentric coordinates within the cell where the probe point lies. The field value is computed as a linear combination of the values at the cell's nodes.
    If the probe point lies outside the source mesh, no interpolation is performed, and the vtkValidPointMask array marks the result as invalid.

    Args:
        comsol_data (pv.PolyData): Source mesh
        control_mesh (vtk.vtkImageData): Control mesh

    Returns:
        pv.ImageData: Mapped data on control mesh.
    """    
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(control_mesh)        # The grid where you want data
    probe.SetSourceData(comsol_data)        # The mesh with the data to interpolate
    probe.Update()
    interpolated = probe.GetOutput()
    return pv.wrap(interpolated)