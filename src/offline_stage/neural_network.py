from typing import List
import torch
from torch import nn


class NIRB_NN(nn.Module):
    """Pytorch Neural Network
    """    
    def __init__(self, n_inputs: int,
                 hidden_units: List[int],
                 n_outputs: int,
                 activation = nn.Sigmoid()):
        super().__init__()
        assert len(hidden_units) > 0, "hidden_units must contain at least one layer"
        
        all_layers = []
        for hidden_unit in hidden_units:
            layer = nn.Linear(n_inputs, hidden_unit)
            all_layers.append(layer)
            all_layers.append(activation)
            n_inputs = hidden_unit
    
        output_layer = nn.Linear(hidden_units[-1], n_outputs)
        all_layers.append(output_layer) 
        
        self.layers = nn.Sequential(*all_layers)        


    def forward(self, x):
        # Flatten all dimensions except the batch dimension
        x  = torch.flatten(x, start_dim=1)
        logits = self.layers(x)
        return logits
    

def get_n_outputs(trained_model) -> int: 
    last_linear = None
    for layer in trained_model.model.layers:
        if isinstance(layer, torch.nn.Linear):
            last_linear = layer

    if last_linear:
        return last_linear.out_features
    else:
        return -1