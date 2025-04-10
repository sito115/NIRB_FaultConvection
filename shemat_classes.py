from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np

def C2K(T: float) -> float:
    return T + 273.15
def K2C(T: float) -> float:
    return T - 273.15


class SHEMAT():
    
    def __init__(self, param_path_rhof: Path, param_path_viscf : Path):
        assert param_path_rhof.exists()
        assert param_path_viscf.exists()
        self.param_rhof  = pd.read_csv(param_path_rhof ,  sep= ' ' , header = None, index_col=0)
        self.param_viscf = pd.read_csv(param_path_viscf , sep=' ',   header = None, index_col=0)
    
    def rhof(self, T: Union[float, np.ndarray], p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Valid for 0.001 - 20 MPa and 15 - 36O degC

        Args:
            T (Union[float, np.ndarray]): [K]
            p (Union[float, np.ndarray]): [Pa]
        """        
        param = self.param_rhof

        T = np.clip(T, C2K(15), C2K(320))
        p = np.clip(p, 0.001e6, 110e6)

        # T in K, p in Pa
        ta = (param.loc['Y0'].item() + param.loc['Y1'].item()*p*1e-6 + param.loc['Y2'].item()*(p*1e-6)**2 + param.loc['Y3'].item()*(p*1e-6)**3 + param.loc['Y4'].item()*(T-273.15) + param.loc['Y5'].item()*(T-273.15)**2 + param.loc['Y6'].item()*(T-273.15)**3 + param.loc['Y7'].item()*(T-273.15)*(p*1e-6) + param.loc['Y8'].item()*(T-273.15) * (p*1e-6)**2 + param.loc['Y9'].item()*(T-273.15)**2 * (p*1e-6) )  
        tb = (param.loc['Z0'].item() + param.loc['Z1'].item()*p*1e-6 + param.loc['Z2'].item()*(p*1e-6)**2 + param.loc['Z3'].item()*(p*1e-6)**3 + param.loc['Z4'].item()*(T-273.15) + param.loc['Z5'].item()*(T-273.15)**2 + param.loc['Z6'].item()*(T-273.15)**3 + param.loc['Z7'].item()*(T-273.15)*(p*1e-6) + param.loc['Z8'].item()*(T-273.15) * (p*1e-6)**2 + param.loc['Z9'].item()*(T-273.15)**2 * (p*1e-6) )
        return ta/tb

    def viscf(self, T: Union[float, np.ndarray], p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Valid for 0.001 - 110 MPa and 15 - 36O degC

        Args:
            T (Union[float, np.ndarray]): [K]
            p (Union[float, np.ndarray]): [Pa]

        Returns:
            _type_: _description_
        """        
        param = self.param_viscf
        T = np.clip(T, C2K(15), C2K(320))
        p = np.clip(p, 0.001e6, 110e6)

        ta = (param.loc['Y0'].item() + param.loc['Y1'].item()*p*1e-6 + param.loc['Y2'].item()*(p*1e-6)**2 + param.loc['Y3'].item()*(p*1e-6)**3 + param.loc['Y4'].item()*(T-273.15) + param.loc['Y5'].item()*(T-273.15)**2 + param.loc['Y6'].item()*(T-273.15)**3 + param.loc['Y7'].item()*(T-273.15)*(p*1e-6) + param.loc['Y8'].item()*(T-273.15) * (p*1e-6)**2 + param.loc['Y9'].item()*(T-273.15)**2 * (p*1e-6) )  
        tb = (param.loc['Z0'].item() + param.loc['Z1'].item()*p*1e-6 + param.loc['Z2'].item()*(p*1e-6)**2 + param.loc['Z3'].item()*(p*1e-6)**3 + param.loc['Z4'].item()*(T-273.15) + param.loc['Z5'].item()*(T-273.15)**2 + param.loc['Z6'].item()*(T-273.15)**3 + param.loc['Z7'].item()*(T-273.15)*(p*1e-6) + param.loc['Z8'].item()*(T-273.15) * (p*1e-6)**2 + param.loc['Z9'].item()*(T-273.15)**2 * (p*1e-6) )
        return ta/tb
    
if __name__ == "__main__":
    shemat = SHEMAT(param_path_rhof =Path('SHEMAT/SHEMAT_param_rhof.csv'),
                    param_path_viscf=Path('SHEMAT/SHEMAT_param_viscf.csv'))
    print(shemat.rhof(300, 1e6))
    print(shemat.viscf(300, 1e6))