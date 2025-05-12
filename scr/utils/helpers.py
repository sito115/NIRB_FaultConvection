import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from pathlib import Path

def mse(predictions :np.ndarray , targets: np.ndarray) -> float:
    """Compute the Mean Squared Error (MSE) between predictions and targets.

    Args:
        predictions (np.ndarray): _description_
        targets (np.ndarray): _description_

    Returns:
        float: _description_
    """    
    return np.mean((predictions - targets)**2)    
    
    
def plot_data(data: np.ndarray, **kwargs):
    """_summary_

    Args:
        data (np.ndarray): _description_
        title (str): _description_
        export_path (Path): _description_
    """    
    fig, ax = plt.subplots(figsize = (12, 4))
    n_points = np.arange(data.shape[1])
    for array in data:
        ax.step(n_points,
                array,
                alpha = 0.5,
                linewidth = 0.5)
    title_string = kwargs.pop("title", "")
    export_path : Path = kwargs.pop("export_path", None)
    ax.set_xlabel("Point ID")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(title_string)
    ax.grid()
    if export_path is not None:
        assert export_path.parent.exists()
        fig.savefig(export_path)
    plt.close(fig)
        

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