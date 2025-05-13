import numpy as np
from typing import Protocol

class Normalizer(Protocol):
    def normalize(self, data : np.ndarray, keep_scaling_params: bool) -> np.ndarray:
        """Normalize the input data."""
        ...
    
    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """Inverse normalize the input data."""
        ...


class MinMaxNormalizer:
    """
    Scales the input data to the range [0, 1] using min-max normalization.
    """  
    def __init__(self):
        self.scaling_params = None
        
    def normalize(self, data: np.ndarray, keep_scaling_params: bool = False) -> np.ndarray:
        if keep_scaling_params:
            min_val = np.min(data)
            max_val = np.max(data)
            scaled_data = min_max_scaler(data, min_val=min_val, max_val=max_val)
            self.scaling_params = {"min_val" : min_val, "max_val" : max_val}
        else:
            scaled_data = min_max_scaler(data, **self.scaling_params)
            
        return scaled_data
        
    def inverse_normalize(self, data) -> np.ndarray:
        if self.scaling_params is None:
            raise ValueError("No scaling paramters kept. Consider the flag 'keep_scaling_params'.")
        return inverse_min_max_scaler(data, **self.scaling_params)

class MeanNormalizer:
    """
    Normalizes the input data using mean normalization, scaling it to have a mean of 0 
    and a range typically between -1 and 1.
    """
    def __init__(self):
        self.scaling_params = None

    def normalize(self, data: np.ndarray, keep_scaling_params: bool = False) -> np.ndarray:
        if keep_scaling_params:
            min_val = np.min(data)
            max_val = np.max(data)
            mean_val = np.mean(data)
            scaled_data = mean_normalization(data, min_val=min_val, max_val=max_val, mean_val=mean_val)
            self.scaling_params = {"min_val" : min_val, "max_val" : max_val, "mean_val": mean_val}
        else:
            scaled_data = mean_normalization(data, **self.scaling_params)
        return scaled_data
    
    def inverse_normalize(self, data) -> np.ndarray:
        if self.scaling_params is None:
            raise ValueError("No scaling paramters kept. Consider the flag 'keep_scaling_params'.")
        return inverse_mean_normalization(data, **self.scaling_params)
    



class Standardizer:
    def __init__(self):
        self.scaling_params = None
        
    def normalize(self, data: np.ndarray, keep_scaling_params: bool = False) -> np.ndarray:
        if keep_scaling_params:
            mean = np.mean(data, axis=0)
            var = np.var(data, axis=0)
            self.scaling_params = {"mean": mean, "var": var}
            scaled_data = standardize(data, mean, var)
        else:
            scaled_data = standardize(data, **self.scaling_params)
            
        return scaled_data
    
    def inverse_normalize(self, data):
        raise NotImplementedError("Not implemented yet")



def min_max_scaler(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Scales the input data to the range [0, 1] using min-max normalization.

    This function applies the min-max normalization technique to scale the input data
    so that it lies within the range [0, 1], using the provided minimum and maximum values.

    Args:
        data (np.ndarray): The input data to be scaled, represented as a NumPy array or array-like structure.
        min_val (float): The minimum value used for scaling the data.
        max_val (float): The maximum value used for scaling the data.

    Returns:
        np.ndarray: The normalized data scaled to the range [0, 1].

    Raises:
        ValueError: If `min_val` is equal to `max_val`, resulting in a division by zero during scaling.
    """
    if min_val == max_val:
        raise ValueError("min_val and max_val cannot be equal, as it leads to division by zero.")
    
    return (np.asarray(data) - min_val) / (max_val - min_val)



def inverse_min_max_scaler(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Inverse min-max scaler to scale the data back to its original range.

    Args:
        data (np.ndarray): Data to be scaled back.
        min_val (float): Minimum value of the original data.
        max_val (float): Maximum value of the original data.

    Returns:
        np.ndarray: Scaled back data.
    """
    data = np.asarray(data)
    return data * (max_val - min_val) + min_val


def mean_normalization(data, min_val: float, max_val: float, mean_val) -> np.ndarray:
    """
    Normalizes the input data using mean normalization, scaling it to have a mean of 0 
    and a range typically between -1 and 1.

    The normalization is performed using the formula:
        (x - mean) / (max - min)

    Where:
        - x: Data value
        - mean: Mean value of the data (or provided mean_val)
        - min: Minimum value of the data (or provided min_val)
        - max: Maximum value of the data (or provided max_val)

    Args:
        data (array-like): Input data to normalize, such as a list, NumPy array, or similar.
        min_val (float): Minimum value of the data.
        max_val (float): Maximum value of the data.
        mean_val (float): Mean value of the data.

    Returns:
        np.ndarray: The normalized data, where each value is transformed based on the formula.

    Raises:
        ValueError: If `max_val` equals `min_val`, leading to a division by zero error.
    """
    
    range_val = max_val - min_val
    if range_val == 0:
        raise ValueError("Normalization range cannot be zero (max_val equals min_val).")
    
    return (np.asarray(data) - mean_val) / range_val
    


def inverse_mean_normalization(data, min_val: float, max_val: float, mean_val: float):
    """
    Reverses mean normalization and reconstructs the original data.

    Args:
        data (array-like): Mean-normalized data.
        min_val (float): Minimum value used during normalization.
        max_val (float): Maximum value used during normalization.
        mean_val (float): Mean value used during normalization.

    Returns:
        np.ndarray: Reconstructed original data.
    
    Raises:
        ValueError: If max_val equals min_val, leading to invalid scaling.
    """
    data = np.asarray(data)
    return data * (max_val - min_val) + mean_val


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