
from pathlib import Path
import pandas as pd
import pint_pandas 
import pint
import numpy as np
from typing import Union

# Patch missing unit BEFORE using .quantify
ureg = pint.UnitRegistry()
ureg.define('radian = [angle] = rad')  # Fix the missing 'rad' (from COMSOL)
PREFERRED_UNITS = {
ureg.Quantity(1, 'degree').dimensionality: 'degree',
ureg.Quantity(1, 'degC').dimensionality: 'degC',
ureg.Quantity(1, 'bar').dimensionality: 'bar',
}
# Register globally for pint-pandas
pint_pandas.PintType.ureg = ureg
pint.set_application_registry(ureg)


def load_pint_data(path: Path, is_numpy = False, **kwargs) -> Union[pd.DataFrame, np.ndarray]:
    """Load csv that has parameter names in the first row and units in the second row.

    Args:
        path (Path): csv-file

    Returns:
        pd.DataFrame: 
    """    
    header = kwargs.pop("header", [0, 1])
    level = kwargs.pop("level", -1)
    training_param = pd.read_csv(path, header=header)
    training_param =  training_param.pint.quantify(level = level)
    if is_numpy:
        return training_param.pint.dequantify().to_numpy()
    return training_param


def convert_to_preferred_unit(q: pint.Quantity) -> pint.Quantity:
    unit = PREFERRED_UNITS.get(q.dimensionality, q.units)
    try:
        q = q.to(unit)
    except pint.errors.UndefinedUnitError:
        pass  # Skip if conversion fails
    return q


def format_quantity(q: pint.Quantity, number_format: str = '.2e') -> str:
    """Display a pint.Quanitity as string in "value unit" format.
    Additionally, preferred units are inserted for temperature, angles.

    Args:
        q (pint.Quantity): 

    Returns:
        str: e.g. "2.02e01 K"
    """    
    q = convert_to_preferred_unit(q)
    return f"{format(q.magnitude, number_format)} {q.units:~P}"

def safe_parse_quantity(s):
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
    
    
def convert_str_to_pint(value: str) -> pint.Quantity:
    """ Converts Comsol parameters to pint.Quantities.

    Args:
        value (str): Comsol parameter value (format is "Value[Unit]")

    Returns:
        pint.Quantity:
    """
    # ureg = pint.UnitRegistry()
    try:
        if "[" in value:
            splitted_value = value.split("[") 
            numeric_value = float(splitted_value[0])
            unit = splitted_value[1].split("]")[0]
            return ureg.Quantity(numeric_value, unit).to_base_units()
        else:
            return float(value) * ureg("dimensionless")
    except ValueError:
        return value  # Return the original value if conversion fails