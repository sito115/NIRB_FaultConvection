import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd
import pint
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parents[1]))
from comsol_module.comsol_classes import COMSOL_VTU
from src.utils import safe_parse_quantity, read_config, PREFERRED_UNITS

ureg = pint.get_application_registry()


def compute_ra(rho0: pint.Quantity,
               c_p: pint.Quantity,
               beta: pint.Quantity,
               delta_T: pint.Quantity,
               k0: pint.Quantity,
               H_reference : pint.Quantity,
               mu: pint.Quantity,
               lambda_m: pint.Quantity,
               g: pint.Quantity) -> pint.Quantity:
    """Computes Rayleigh number of a fault zone.

    Eq. 14 in Zhao et al. (2004): “Theoretical Investigation of
    Convective Instability in Inclined and Fluid-Saturated
    Three-Dimensional Fault Zones.”


    Args:
        rho0 (pint.Quantity): Reference density.
        c_p (pint.Quantity): specific heat of the pore fluid.
        beta (pint.Quantity): thermal volume expansion coefficient of the pore-fluid.
        delta_T (pint.Quantity): temperature difference between the bottom and top boundaries of the inclined fault zone;
        k0 (pint.Quantity): permeability of the inclined fault zone
        H_reference (pint.Quantity): Reference length.
        mu (pint.Quantity): dynamic viscosity of the pore fluid.
        lambda_m (pint.Quantity): bulk thermal conductivity of fault zone.
        g (pint.Quantity): gravitational force 

    Returns:
        pint.Quantity: Rayleigh numnber.
    """
    ra = rho0**2 * c_p * g * beta * delta_T * k0 * H_reference / (mu * lambda_m)
    assert ra.check(['dimensionless']) # check for correct unit
    return ra

def main():
    
    ROOT = Path().cwd()
    PARAMETER_SPACE = "09"
    DATA_TYPE = "Test"
    config = read_config()
    x_axis_name = "dip"
    IS_EXPORT = True

    import_folder = ROOT / "data" / PARAMETER_SPACE /  f"{DATA_TYPE}Original"
    export_folder = ROOT / "data" / PARAMETER_SPACE / "Exports"
    assert import_folder.exists()
    assert export_folder.exists()
    comsol_vtu_files = sorted([path for path in import_folder.rglob("*.vt*")])
    N_SNAPS = len(comsol_vtu_files)

    param_folder = ROOT / "data" / PARAMETER_SPACE / "Exports"
    param_files = sorted([path for path in param_folder.rglob(f"{DATA_TYPE}*.csv")])
    assert len(param_files) == N_SNAPS
    
    ra_numbers = np.zeros((N_SNAPS, ))
    x_vals = np.zeros_like(ra_numbers)
    for idx_snap in tqdm(range(N_SNAPS), total = N_SNAPS):
        comsol_data = COMSOL_VTU(comsol_vtu_files[idx_snap],  is_clean_mesh=False)
        for field_name in ["Fluid_density", "Dynamic_viscosity", "Temperature"]:
            assert field_name in comsol_data.exported_fields
        param_df = pd.read_csv(param_files[idx_snap], index_col = 0)
        param_df['quantity_pint'] = param_df[param_df.columns[-1]].apply(lambda x : safe_parse_quantity(x))
        lambda_bulk = (1 - param_df.loc['fault_phi', "quantity_pint"]) * param_df.loc['fault_lambda', "quantity_pint"] + \
                            param_df.loc['fault_phi', "quantity_pint"] * (config['lambda_f']  * ureg.watt / (ureg.meter * ureg.kelvin))
        delta_T = param_df.loc['T_h', "quantity_pint"]  - param_df.loc["T_c", "quantity_pint"]
        dip = param_df.loc['dip', "quantity_pint"]
        H = param_df.loc['H', "quantity_pint"]
        k = param_df.loc['fault_k_trans', "quantity_pint"]
        h_reference = H / np.sin(dip)
        fluid_density = comsol_data.get_point_values("Fluid_density", 0)
        rho0 = np.mean(fluid_density) * ureg.kg / ureg.meter**3
        mu = np.mean(comsol_data.get_point_values("Dynamic_viscosity", 0)) * ureg.pascal * ureg.second
        
        assert x_axis_name in param_df.index
        temp_val = param_df.loc[x_axis_name, "quantity_pint"]
        unit = PREFERRED_UNITS[temp_val.dimensionality]
        x_vals[idx_snap] = temp_val.to(unit).magnitude    
        
        x_locs = np.array([1/3, 2/3]) * comsol_data.mesh.bounds[1] #xmax
        y_locs = np.array([1/3, 2/3]) * comsol_data.mesh.bounds[3] #ymax
        betas = []
        resolution = 100
        field_name_density = comsol_data.format_field("Fluid_density", 0)
        field_name_temperature = comsol_data.format_field("Temperature", 0)
        for x_loc in x_locs:
            for y_loc in y_locs:
                pointa = [x_loc, y_loc, comsol_data.mesh.bounds.z_max]
                pointb = [x_loc, y_loc, comsol_data.mesh.bounds.z_min]
                sampled_mesh = comsol_data.mesh.sample_over_line(pointa, pointb, resolution = resolution)
                
                temp_beta  = np.nanmean( - np.gradient(np.log(sampled_mesh.point_data[field_name_density]), sampled_mesh.point_data[field_name_temperature]))  # Eq. 12 in Börsing (2017)
                betas.append(temp_beta)
        beta = np.mean(np.array(betas)) / ureg.kelvin
        
        g = 9.81 * ureg.meter / ureg.second**2
        # g = g * np.sin(dip)
        
        ra_num = compute_ra(H_reference=h_reference,
                        delta_T=delta_T,
                        rho0=rho0,
                        lambda_m=lambda_bulk,
                        k0 = k,
                        mu = mu,
                        beta=beta,
                        c_p = config["c_pf"] * ureg.joule / (ureg.kg * ureg.kelvin),
                        g = g
                        )

        ra_numbers[idx_snap] = ra_num
    
    if IS_EXPORT:
        np.save(export_folder / f"{DATA_TYPE}_rayleigh_numbers.npy", ra_numbers)
    print(ra_numbers)
    print("Exported Rayleigh Numbers")
    
    fig, ax = plt.subplots()
    ax.scatter(x_vals, ra_numbers)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel("Rayleigh number")
    ax.set_title(f"{DATA_TYPE} - PS{PARAMETER_SPACE}")
    if IS_EXPORT:
        fig.savefig(export_folder / f"{DATA_TYPE}_rayleigh_numbers.png")
    plt.show()
    

if __name__ == "__main__":
    main()