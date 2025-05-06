import numpy as np
from sklearn.metrics import mean_squared_error

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



if __name__ == "__main__":
    pass