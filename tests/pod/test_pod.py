import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from src.pod import POD



def test_pod_time() -> None:
    """Test implementation of POD with the one from https://zenodo.org/records/8369108 / geothermal example.
    Tolerance at machine episolon.
    """    
    root = Path(__file__).parent
    training_snapshots = np.load(root / "TrainingSnapshots.npy")
    expected_basis_function = np.load(root / "BasisFunctionsGeothermalTemperature.npy")
    temperatures = training_snapshots[1]
    accuracy = 1e-5
    pod = POD(temperatures, is_time_dependent=True)
    basis_functions, _ = pod.perform_POD(accuracy)
    npt.assert_allclose(basis_functions, expected_basis_function, atol=1e-16)
    
    
def test_pod_stationary() -> None:
    """implementation of POD with the one from https://zenodo.org/records/13767010 / Benchmark / Stress XX
    Tolerance at machine episolon.
    """    
    root = Path(__file__).parent
    training_snapshots = np.load(root / "TrainingDataBoundaryConditionsStressXX_scaled_borehole.npy")
    expected_basis_function = np.load(root / "BoundaryConditionsStressXXBasisFts.npy")
    stress = training_snapshots
    accuracy = 1e-6
    pod = POD(stress[0], is_time_dependent=False)
    basis_functions, _ = pod.perform_POD(accuracy)
    npt.assert_allclose(basis_functions, expected_basis_function[0], atol=1e-16)

    
if __name__ == "__main__":
    test_pod_stationary()