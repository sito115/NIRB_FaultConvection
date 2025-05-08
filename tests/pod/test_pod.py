import pytest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from scr.pod import POD



def test_pod() -> None:
    root = Path(__file__).parent
    training_snapshots = np.load(root / "TrainingSnapshots.npy")
    expected_basis_function = np.load(root / "BasisFunctionsGeothermalTemperature.npy")
    temperatures = training_snapshots[1]
    accuracy = 1e-5
    pod_t = POD(temperatures, is_time_dependent=True)
    basis_functions, _ = pod_t.perform_POD(accuracy)
    npt.assert_allclose(basis_functions, expected_basis_function, atol=1e-15)
    
if __name__ == "__main__":
    test_pod()