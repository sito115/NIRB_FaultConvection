from attr import field
import scipy.constants
from pydantic import BaseModel, BeforeValidator , Field, field_validator
from pydantic.dataclasses import dataclass
from typing import Optional
import numpy as np  
from pathlib import Path
from typing import Union, Annotated, List
from enum import Enum
import pyvista as pv
import logging
import re
from tqdm import tqdm
# from numba import jit


class ModelData(BaseModel):
    alpha: Optional[float] = None
    rho0: Optional[float] = None
    c_f: Optional[float] = None
    T_c: Optional[float] = None
    T_h: Optional[float] = None
    H: Optional[float] = None
    mu0: Optional[float] = None
    lambda_m: Optional[float] = None  # Allow lambda_m to be a part of the model for easy access after calculation
    k_m: Optional[Union[float, np.ndarray]] = None 
    
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like np.ndarray
    
    
    def calculate_lambda_m(self, lambda_f, lambda_s, phi) -> float:
        """ 
        Calculate the effective thermal conductivity of the porous medium (lambda_m).

        Args:
            lambda_f (float): Thermal conductivity of the fluid [W/mK].
            lambda_s (float): Thermal conductivity of the solid phase [W/mK].
            phi (float): Porosity of the material [-].

        Returns:
            float: Effective thermal conductivity of the porous medium [W/mK].

        """
        self.lambda_m = phi * lambda_f + (1 - phi) * lambda_s
        
        return self.lambda_m
    
    def calculate_T0(self) -> float:
        return 0.5 * (self.T_c + self.T_h) 


    def calculate_rayleigh_number(self) -> float:
        """
        Calculate the Rayleigh number for convection in a porous medium.
        - alpha = Thermal expansion coefficient [1/°C]
        - rho0 = Density of the fluid [kg/m3]
        - c_f = Specific heat of the fluid [J/kg/°C]
        - g = Gravitational constant [m2/s]
        - delta_T = Temperature difference [K]
        - K_m = Permeability of the porous material [m2]
        - H = Box size [m]
        - mu0 = Viscosity of the fluid [Pa/s]
        - lambda_m = Effective thermal conductivity of the porous 

        Returns:
            float: Rayleigh number [-].
        """
        
        for key, val in iter(self):
            if val is None:
                raise ValueError(f"Missing value for {key}")
        
        rayleigh_number =  self.k_m * self.rho0**2 * self.c_f * scipy.constants.g * self.alpha * (self.T_h - self.T_c) *  self.H / (self.mu0 * self.lambda_m) # 
        
        return rayleigh_number
        

# @jit
def calculate_S_therm(lambda_m : float, T_0 : float, temp_gradient : np.ndarray) -> np.ndarray:
    """Calculate the thermal entropy generation rate per volume.

    Args:
        lambda_m (float): _description_
        T_0 (float): _description_
        temp_gradient (np.ndarray): [N x 3] matrix of temperature gradient components [K/m]

    Returns:
        np.ndarray: entropy generation rate per VOLUME [W/(K * m^3 * s)]
    """
    return lambda_m / T_0**2 * (temp_gradient[:, 0]**2 + temp_gradient[:, 1]**2 + temp_gradient[:, 2]**2) 

# @jit
def calculate_S_visc(mu_f : float, k_tensor : np.ndarray, T_0 : Union[float, np.ndarray], darcy_vel : np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        mu_f (float): _description_
        k_tensor (np.ndarray): _description_
        T_0 (float): _description_
        darcy_vel (np.ndarray): [3 x N] matrix of darcy velocity components [m/s]

    Returns:
        np.ndarray: entropy generation rate per VOLUME [W/(K * m^3 * s)]
    """
    
    return mu_f / (np.mean(k_tensor) * T_0) * (darcy_vel[0]**2 + darcy_vel[1]**2 + darcy_vel[2]**2) 



def calculate_S_total(temp_gradient : np.ndarray, darcy_vel : np.ndarray,  T_0 : Union[float, np.ndarray], lambda_m : float, mu_f : float, k_tensor : np.ndarray) -> np.ndarray:
    """Calculate the total entropy generation rate per volume composed of viscous and thermal term.

    Args:
        temp_gradient (np.ndarray): [N x 3] matrix of temperature gradient components [K/m]
        darcy_vel (np.ndarray): [3 x N] matrix of darcy velocity components [m/s]
        T_0 (float): _description_
        lambda_m (float): _description_
        mu_f (float): _description_
        k_tensor (np.ndarray): _description_

    Returns:
        np.ndarray: Tototal entropy generation rate per VOLUME [W/(K * m^3 * s)]
    """  
    
    S_therm = calculate_S_therm(lambda_m , T_0 , temp_gradient)
    S_visc  = calculate_S_visc(mu_f, k_tensor, T_0 , darcy_vel)
    
    return S_therm + S_visc

# @jit
def caluclate_entropy_gen_number(s_total : Union[float, np.ndarray], q_field: float, lambda_m :float, T_0: float, V: float):
    """_summary_

    Args:
        S_total (_type_): total entropy generation rate [W/(K * s)]
        q_field (_type_):     q is the total specific heat flow, represents the bulk heat flux between the lower and upper boundaries of the model [W/m^2]
        lambda_m (_type_): bulk thermal conductivity of the porous medium [W/mK]
        T_0 (_type_): _description_
        V (_type_): Volume of the porous medium [m^3]

    Returns:
        _type_: _description_
    """  
    return s_total * lambda_m * T_0**2 / (q_field**2 * V )

class ComsolKeyNames(Enum):
    "Temperature_@_t={key}"
    T = 'Temperature' 
    T_grad_x = 'Temperature_gradient,_x-component'
    T_grad_y = 'Temperature_gradient,_y-component'
    T_grad_z = 'Temperature_gradient,_z-component'
    darcy_x = 'Total_Darcy_velocity_field,_x-component'
    darcy_y = 'Total_Darcy_velocity_field,_y-component'
    darcy_z = 'Total_Darcy_velocity_field,_z-component'
    s_tol = 'Total_Entropy_CellBased'


def compute_surface_normal_vector(bounds: pv.DataSet.bounds) -> np.ndarray:
    """Computes surface normal vector from the bounds of the 2D surface.

    Args:
        bounds (pv.DataSet.bounds): _description_

    Returns:
        np.ndarray: _description_
    """        
    point1 = np.array([bounds[1], bounds[2], bounds[4]]) # base point bottom
    point2 = np.array([bounds[1], bounds[3], bounds[4]]) # lateral extent (y-dir)
    point3 = np.array([bounds[0], bounds[2], bounds[5]]) # vertical extent (z-dir)
    vector1 = point2 - point1  
    vector2 = point3 - point1  
    return  np.cross(vector1, vector2), point1 # normal vector from crossproduct


def ensure_pathlib_path(path: Union[str,Path, List]) -> Union[List[Path], Path]:
    if isinstance(path, List):
        return [Path(v) if not isinstance(v, Path) else v for v in path]
    else:
        return Path(path) if isinstance(path, str) else path

def initilise_plotter(mesh: pv.DataSet, mp4_file: Path) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(mp4_file)
    plotter.add_mesh(mesh.outline_corners())
    plotter.add_axes()
    plotter.show_bounds(mesh)
    return plotter

def read_comsol_fields(mesh:pv.DataSet, field_pattern, time_pattern) -> tuple[list[str], dict[str, float]]:
    """Field names in COMSOL are FIELDNAME_@_tTIME.

    Args:
        mesh (pv.DataSet): 
        field_pattern (_type_): regex to find field names in pyvista dataset,
                               
        time_pattern (_type_): regex to find time in pyvista dataset,

    Returns:
        tuple[pv.DataSet,list[str], dict[str, float]]: _description_
    """    
    exported_fields : list[str] = list(set([re.search(field_pattern, key).group(1) for key in mesh.point_data.keys()]))
    # Sort the times and map them back to the original string values
    time_map : dict[str:float] = {re.search(time_pattern, key).group(1): float(re.search(time_pattern, key).group(1)) for key in mesh.point_data.keys()}
    times : dict[str:float]= dict(sorted(time_map.items(), key=lambda x: x[1]))  # Sort by float value
    return (exported_fields, times)    

@dataclass
class COMSOL_VTU():
    """_summary_

    Raises:
        ValueError: _description_

    Attributes:
        mesh (pv.DataSet): _description_
        times (dict(str:float)): Sorted dictionary of exported times. The key corresponds to the exact string in the field name.
        exported fields (list(str)): All exported field names.
    """    
    vtu_path: Union[Path, str]
    optional_vtu_paths : Optional[Union[List[Path], Path]] = None
    name : Optional[str] = ''
    vtu_pattern =  '{}_@_t={}'                # container for finding values in pyvista.DataSet

    
    class Config:
        arbitrary_types_allowed = True # for numpy etc
    
    @field_validator("vtu_path")
    @classmethod
    def check_path_exists(cls, vtu_path: Union[Path, str]) -> Path:#
        vtu_path = ensure_pathlib_path(vtu_path)
        if not vtu_path.exists():
            raise ValueError(f'Given path does not exist: {vtu_path}.')
        return vtu_path
    
    @field_validator("optional_vtu_paths")
    @classmethod
    def check_opt_path_exists(cls, vtu_path: Union[List[Path], Path]) -> List[Path]:#
        if vtu_path is None:
            return None
        vtu_path = ensure_pathlib_path(vtu_path)
        if not isinstance(vtu_path, list):
            vtu_path = [vtu_path]
        if not all([path.exists() for path in vtu_path]):
            raise ValueError('Given paths in optional vtu path does not exist')
        return vtu_path 
    
    def __post_init__(self):
        logging.debug('Reading vtu file...')
        self.mesh = pv.read(self.vtu_path)
        logging.debug('Finished')
        time_pattern = r"@_t=([\d.]+(?:[Ee][+-]?\d+)?)" # find exported time steps
        field_pattern = r"^(.*?)_@_t"                    # find exported fields
        self.exported_fields, self.times = read_comsol_fields(self.mesh, field_pattern, time_pattern)
        if self.optional_vtu_paths is not None:
            for path in self.optional_vtu_paths:
                temp_mesh = pv.read(path)
                temp_exported_fields, temp_times = read_comsol_fields(temp_mesh, field_pattern, time_pattern)
                assert set(temp_exported_fields).issubset(set(self.exported_fields))
                assert self.mesh.points.shape == temp_mesh.points.shape
                self.times.update(temp_times)
                self.mesh.point_data.update(temp_mesh.point_data)

    def info(self):
        print(f'{self.vtu_path=}')
        print(f'{len(self.times)} timesteps from {min(self.times.values()):.3e} s to {max(self.times.values()):.3e} s')
        print(f'{self.mesh.bounds=}')
        print('Availabe fields in vtu dataset:')
        for idx, field in enumerate(sorted(self.exported_fields), start=1):
            print('\t %d: %s' % (idx, field))
        
    def export_mp4_movie(self, field: str, mp4_file: Path = None, **kwargs) -> None:
        """Exports a mp4 movie.

        Args:
            field (str): _description_
            mp4_file (Path, optional): _description_. Defaults to None.
        
        Keyword Args:
            is_diff (bool, optional): Subtract the value at the first time index from every iteration. Defaults to False.
            is_ind_cmap (bool, optional): Display an individual colormap for each iteration. Defaults to False.
            t_grad (float, optional): Temperature gradient in the z-direction that is subtracted from every iteration (in [K/m]). 
            movie_field (str, optional): Name of the field to display above the colormap in the movie.
            bounds (tuple of float): Tuple of the form (xmin, xmax, ymin, ymax, zmin, zmax) for the clipped array.
            add_mesh_kwargs (dict): Additional keyword arguments for `pv.add_mesh()`.
            is_min_max (bool, optional): Display min and max values in the colorbar. Defaults to False.
            
        Returns:
            _type_: None
        """
        movie_field = kwargs.pop('movie_field', field + "-mp4")
        if mp4_file is None:
            mp4_file = self.vtu_path.parent.joinpath(f'{self.vtu_path.stem}_{field}.mp4')
        print('Export path = %s' % str(mp4_file))
        plotter = initilise_plotter(self.mesh, mp4_file)

        # Add the scalar bar to the plotter
        plotter.add_scalar_bar(title=movie_field, label_font_size=12, 
                       position_x=0.2, position_y=0.05)

        bounds = kwargs.pop('bounds', None)
        if bounds is None:
            mesh = self.mesh
        else:
            mesh = self.mesh.clip_box(bounds)
        
        key0 = next(iter(self.times.keys()))
        val0 = mesh[self.vtu_pattern.format(field,key0)]
         
        t_grad = kwargs.pop('t_grad', None)
        if t_grad is not None:
            z = mesh.points[:, 2]
            val0 = t_grad['t0'] - t_grad['t_grad'] * z 
                
        is_diff = kwargs.pop('is_diff', False)
        if is_diff:
            mesh[movie_field] = val0 - val0
        else:
            mesh[movie_field] = val0
        
        add_mesh_kwargs = kwargs.pop('add_mesh_kwargs', {})
        actor = plotter.add_mesh(mesh, scalars = movie_field, **add_mesh_kwargs) # The sliced plane
        plotter.write_frame()
        
        is_ind_cmap = kwargs.pop('is_ind_cmap', False)
        is_min_max = kwargs.pop('is_min_max', False)
        if is_min_max:
            win_width, win_height = plotter.render_window.GetSize()
            x, y = plotter.scalar_bar.GetPosition()
            width, _ = plotter.scalar_bar.GetPosition2()
            min_text_position = ((x - 0.1)*win_width, y*win_height) 
            max_text_position = ((x + width + 0.02)*win_width, y*win_height)
            fs = plotter.scalar_bar.GetLabelTextProperty().GetFontSize()
            # label_format = plotter.scalar_bar.GetLabelFormat()
            min_text_actor = plotter.add_text(f"Min: {np.min(mesh[movie_field]):.2e}", font_size=0.6 * fs,
                             color="black", position=min_text_position)
            max_text_actor = plotter.add_text(f"Max: {np.min(mesh[movie_field]):.2e}", font_size=0.6 * fs,
                             color="black", position=max_text_position)

        for idx, (key, time) in tqdm(enumerate(self.times.items(), start = 1), desc=f'Processing frames for {field}', total = len(self.times)):
            if is_diff:
                mesh[movie_field] = mesh[self.vtu_pattern.format(field,key)] - val0
            else:
                mesh[movie_field] = mesh[self.vtu_pattern.format(field,key)]
            plotter.add_text(f"Output {idx}: {time:.3e} s", name='time-label')
            if is_ind_cmap:
                actor.mapper.scalar_range = ( np.min(mesh[movie_field]), np.max(mesh[movie_field]) )
            if is_min_max:
                min_text_actor.input = f'Min: {np.min(mesh[movie_field]):.2e}'
                max_text_actor.input = f'Max: {np.max(mesh[movie_field]):.2e}'
            plotter.write_frame()
            
        plotter.close()

    def get_point_values(self, time_step : Union[int, str], field: ComsolKeyNames) -> np.ndarray:
        if isinstance(time_step, int):
            key = list(self.times.keys())[time_step]
            logging.info(f'Time step {key}')
        else:
            key = time_step
        return self.mesh.point_data[self.vtu_pattern.format(field,key)]
    
    def unify_field(self, field_name:str) -> None:
        self.mesh.point_data[field_name] = self.mesh.point_data[self.vtu_pattern.format(field_name, list(self.times.keys())[0])]
        for key in tqdm(self.times.keys(), f"Removing redundant fields '{field_name}'"):
            self.mesh.point_data.remove(self.vtu_pattern.format(field_name, key))
    
    def format_fied(self, field_name: str, time: Union[str, float, int]):
        assert field_name in self.exported_fields
        if isinstance(time, str):
            assert time in self.times.keys()
        if isinstance(time, float):
            time = min(self.times, key=lambda k: abs(self.times[k] - time))
        if isinstance(time, int):
            assert time <= len(self.times)
            time = list(self.times.keys())[time]
        return self.vtu_pattern.format(field_name, time)
    
    def calculate_total_quantity(self, field_name: str, fields: list[str] = None) -> None:
        """Calculates L2 norm of given fields.

        Args:
            fields (list[str]): list of fields to calculate the L2 norm (must be in self.exported_fields)
            field_name (str): the new field name (will be appended to self.exported_fields)
        """        
        
        if fields is None:
            fields = ['Total_Darcy_velocity_field,_x-component',
                    'Total_Darcy_velocity_field,_y-component',
                    'Total_Darcy_velocity_field,_z-component',]
        
        self.exported_fields.append(field_name)
        for time_key in tqdm(self.times.keys(), desc='Processing...', total = len(self.times)): 
            quantities = np.zeros((len(fields), len(self.mesh.points)))
            for i_field, field in enumerate(fields):  # noqa: F402
                quantities[i_field] = self.mesh.point_data[self.vtu_pattern.format(field, time_key)]
            self.mesh.point_data[self.format_fied(field_name, time_key)] = np.sqrt(np.sum(quantities**2, axis = 0))
        
    
    def overwrite_domain_from_surface(self, surface: pv.DataSet, field_name3D: str, field_name2D: str = 'Color') -> None:
        """Can be used, when the 2D surface is a subset of the 3D domain. For example, if a fracture is implemented in Comsol (dl.frac), its Darcy-Velocieties
        from an export in surface-plot will differ from data Export in 3D domain. This function overwrites the 3D domain with the values from the 2D surface.
        Args:
            surface (pv.DataSet): 
            field_name3D (str): Field name in COMSOL_VTU object in mesh.point_data
            field_name2D (str, optional): Field name in Surface DataSet. Defaults to 'Color'.

        Returns:
            _type_: None
        """
        assert field_name3D in self.mesh.point_data.keys()
        assert field_name2D in surface.point_data.keys()
        
        # Structured view helper function for comparing elements in surface and domain row-wise
        def structured_view(arr : np.ndarray) -> np.ndarray:
            return arr.view([('', arr.dtype)] * arr.shape[1])
    
        mask3d = np.isin(structured_view(self.mesh.points), structured_view(surface.points)).flatten()
        mask2d = np.isin(structured_view(surface.points), structured_view(self.mesh.points)).flatten()

        points_3d_masked = self.mesh.points[mask3d, :]
        vals_3d = np.zeros((np.sum(mask3d)))
        for tuple2d, val in zip(surface.points[mask2d, :], surface.point_data[field_name2D][mask2d]):
            mask = np.all(points_3d_masked == tuple2d, axis=1)
            vals_3d[mask] = val

        self.mesh.point_data[field_name3D][mask3d] = vals_3d
    
    
    def get_cell_values(self, time_step :  Union[int, str], field: ComsolKeyNames) -> np.ndarray:
        data = self.mesh.point_data_to_cell_data()
        if isinstance(time_step, int):
            key = list(self.times.keys())[time_step]
            logging.info(f'Time step {key}')
        else:
            key = time_step
        return data.cell_data[self.vtu_pattern.format(field,key)]
    
    def compute_cell_data(self, **kwargs):
        """
            pass_point_databool, default: False, keep point data in pyvista object
        """
        is_pass_point_data = kwargs.pop('pass_point_data', True)        
        self.mesh = self.mesh.point_data_to_cell_data(pass_point_data=is_pass_point_data)
    
    
    def get_array(self, field: ComsolKeyNames, is_cell_data : bool = False) -> np.ndarray:
        """_summary_

        Args:
            field (ComsolKeyNames): _description_
            
        Keyword Args:
            city (str, optional): The user's city.
            country (str, optional): The user's country.
            active (bool, optional): Whether the user is active.

        Returns:
            np.ndarray: _description_
        """
        if is_cell_data:
            return np.array([self.mesh.cell_data[self.format_fied(field, key)] for key in self.times.keys()])        
        return np.array([self.mesh.point_data[self.format_fied(field, key)] for key in self.times.keys()])
            
            
    def calculate_total_entropy_per_vol(self,
                                        model_data: ModelData,
                                        time_steps: Union[list[int],int] = None,
                                        is_save_as_point_data: bool = False) -> np.ndarray:
        """_summary_

        Args:
            model_data (ModelData): _description_
            time_steps (Union[list[int],int]): zero-indexed!

        Returns:
            np.ndarray: [N x 2] Entropy (therm, visc) in  W/(K * s)
        """
        if time_steps is None:
            time_steps = np.arange(len(self.times))
        
        if isinstance(time_steps, int):
            time_steps = [time_steps]
            
        assert max(time_steps) <= len(self.times) and min(time_steps) >= 0
        
        time_keys : list[str] = list(self.times.keys())
        time_keys : list[str] = [time_keys[i] for i in time_steps]
        
        data = self.mesh.point_data_to_cell_data() # convert to cell data to match cell areas to points in gradient
        
        integraded_entropy = np.zeros((len(time_steps), 2))
        for idx, time_key in enumerate(time_keys):
            logging.debug(f'Time {time_key}')
            
            temp_gradient = data.compute_derivative(scalars=self.vtu_pattern.format(ComsolKeyNames.T.value,time_key)).cell_data['gradient']
            try:
                dary_x = data.cell_data[self.vtu_pattern.format(ComsolKeyNames.darcy_x.value,time_key)]
            except KeyError as e:
                logging.warning(f'{e} - Created 0 matrix')
                dary_x = np.zeros_like(temp_gradient[:, 0])
                
            try:
                dary_y = data.cell_data[self.vtu_pattern.format(ComsolKeyNames.darcy_y.value,time_key)]
            except KeyError as e:
                logging.warning(f'{e} - Created 0 matrix')
                dary_y = np.zeros_like(temp_gradient[:, 0])
                
            try:
                dary_z = data.cell_data[self.vtu_pattern.format(ComsolKeyNames.darcy_z.value,time_key)]
            except KeyError as e:
                logging.warning(f'{e} - Created 0 matrix')
                dary_z = np.zeros_like(temp_gradient[:, 0])

            # TODO: replace T0 with temperature field
            s_therm = calculate_S_therm(model_data.lambda_m , model_data.calculate_T0() , temp_gradient)
            s_visc  = calculate_S_visc(model_data.mu0, model_data.k_m, model_data.calculate_T0() , [dary_x, dary_y, dary_z])
            
            if is_save_as_point_data:
                self.mesh.cell_data[self.vtu_pattern.format(ComsolKeyNames.s_tol.value, time_key)] = (s_therm + s_visc).copy()
            
            integraded_entropy[idx, 0] = np.sum(s_therm * data.compute_cell_sizes()['Area']) 
            integraded_entropy[idx, 1] = np.sum(s_visc * data.compute_cell_sizes()['Area'])
            
        return integraded_entropy
    
    def merge_datasets(self, *args) -> None:
        for arg in args:
            assert isinstance(arg, COMSOL_VTU)
            assert set(arg.exported_fields).issubset(set(self.exported_fields))
            assert self.mesh.points.shape == arg.mesh.points.shape
            self.times.update(arg.times)
            self.mesh.point_data.update(arg.mesh.point_data)
            

        
if __name__ == '__main__':
    
    # Set up basic configuration for logging
    logging.basicConfig(
    level=logging.DEBUG,               # Set the lowest level of logging to capture
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the format of log messages
    )
    
    root = Path('MergeSurface2Domain')
    path_fault_2d  = root / 'Fracture_sol1.vtu'
    path_fault_2d.exists()
    path_domain_3d = root / 'Solution1.vtu'
    fault_2d  = pv.read(path_fault_2d)
    domain_3d = COMSOL_VTU(path_domain_3d)

    # plotter.show()
    def create_smooth_bounds(mesh: pv.DataSet, val_old : float, val_new: float, axis: str = 'z') -> pv.DataSet:
        map_axis_dict = {'x': 0, 'y': 1, 'z': 2}
        axis = map_axis_dict[axis]
        mesh.points[:, axis] = np.where(mesh.points[:, axis] == val_old, val_new, mesh.points[:, axis])
        return mesh

    print(domain_3d.mesh.bounds)
    while domain_3d.mesh.bounds[-1] > 0.:
        domain_3d.mesh = create_smooth_bounds(domain_3d.mesh, domain_3d.mesh.bounds[-1], 0. , axis='z')
    print(domain_3d.mesh.bounds)


    print(fault_2d.bounds)
    while fault_2d.bounds[-1] > 0.:
        fault_2d = create_smooth_bounds(fault_2d, fault_2d.bounds[-1], 0. , axis='z')
    print(fault_2d.bounds)
    
    normal_vector, point1 = compute_surface_normal_vector(fault_2d.bounds)
    
    field_name = domain_3d.format_fied(ComsolKeyNames.T.value, -1)
    domain_3d.mesh.clip(normal = normal_vector, origin=point1).plot(scalars = field_name)
    
    bounds = domain_3d.mesh.bounds
    xrng = np.arange(bounds[0], bounds[1] + 1, 100)
    yrng = np.arange(bounds[2], bounds[3] + 1, 100)
    zrng = np.arange(bounds[4], bounds[5] + 1, 100)
    grid = pv.RectilinearGrid(xrng, yrng, zrng)
    grid.plot(show_edges=True)
    print(grid.bounds)
    print(domain_3d.mesh.bounds)
    
    
    result              = grid.sample(domain_3d.mesh, categorical=True)
    clip = result.clip(normal = normal_vector, origin=point1)
    # Visualize
    clip.plot(scalars=field_name, cmap="coolwarm", show_edges=True)