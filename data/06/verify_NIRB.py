"""Verify the code with example from "3D multi-physics uncertainty quantification using physics-based machine learning".
Surrogate Model Groß Schöneback.
Original code available at 
"""
from matplotlib import pyplot as plt
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
import numpy as np
from pathlib import Path
import sys
import logging
sys.path.append(str(Path(__file__).parents[2]))
from src.offline_stage import NirbDataModule, NirbModule, Normalizations
from src.pod import POD
from src.utils import setup_logger

def main(is_train: bool = False) -> NirbDataModule:
    seed_everything(20112020)
    root = Path(__file__).parent
    training_snaps = np.load(root / 'GrSbk_snapshots.npy')[0, :, :]
    training_param = np.load(root / 'GrSbk_TrainingParameters.npy')
    test_param     = np.load(root / 'GrSbk_ValidationParameters.npy')
    test_snaps     = np.load(root / 'GrSbk_val_snapshots.npy')[0, :, :]
    
    
    pod_class = POD(training_snaps, is_time_dependent=False)
    bsf_functions , info = pod_class.perform_POD(accuracy=1e-6)
    logging.debug(info)
    
    data_module = NirbDataModule(training_snaps=training_snaps,
                                 training_param=training_param,
                                 test_snaps=test_snaps,
                                 test_param=test_param,
                                 basis_func_mtrx=bsf_functions,
                                 batch_size=25,
                                 normalizer= Normalizations.NoNormalization,
                                 standardizer_features=Normalizations.Standardizer)
    
    if is_train:
    
        model = NirbModule(training_param.shape[1],
                            [15, 25, 30, 35, 15],
                            bsf_functions.shape[0],
                            activation=torch.nn.Sigmoid(),
                            learning_rate=1e-03)

        logger = TensorBoardLogger(root)
        trainer = L.Trainer(max_epochs=40_000,
                        strategy='ddp',
                        logger=logger,
                        enable_progress_bar=False,
                        devices=1,
                        accelerator= "cpu", #'mps',
                        log_every_n_steps = 25
                        )
        
        trainer.fit(model=model,
                    train_dataloaders=data_module.train_dataloader(shuffle = True))
         
        model.basis_functions = data_module.basis_func_mtrx
        model.test_snaps_scaled = data_module.test_snaps_scaled
        
        results = trainer.test(model=model,
                                dataloaders=data_module.test_dataloader())
        
        print(results)
        
    return data_module


def online_stage(mu_online : np.ndarray, model_ckpt_path : Path, bsf_functions: np.ndarray):
    """This function executes the online stage of the non-instrusive RB method.

    Args:
        mu_online = parameters for the online solve
        number_of_variables = number_of_variables

    """
    trained_model : NirbModule = NirbModule.load_from_checkpoint(model_ckpt_path)
    trained_model = trained_model.to('cpu')
    trained_model.eval()
    print('Predicting the reduced coefficients')
    mu_online_t = torch.from_numpy(mu_online.astype(np.float32)).view(1, -1) # shape [1, n_param]
    rb_coeff = trained_model(mu_online_t)
    rb_coeff_np = rb_coeff.detach().numpy()
    print(rb_coeff)

    print('Projecting the reduced solution onto the high-dimensional space')
    full_solution = np.dot(rb_coeff_np.flatten(), bsf_functions)
    return rb_coeff_np, full_solution





if __name__ == "__main__":
    setup_logger(is_console=True)
    data_module = main(is_train=False)
    
    # %% Example: Online stage
    # Define online parameters
    mu_online = np.array([[5.51590170e+10, 4.74981480e-07, 3.64472213e-16, 1.94969970e-01,
                        1.00768817e-16, 2.57007007e-03]])

    mu_online_scaled = data_module.standardizer.normalize_reuse_param(mu_online)

    root = Path(__file__).parent 

    rb_coeff_np, full_solution = online_stage(mu_online_scaled,
                                 root / "lightning_logs/version_3/checkpoints/epoch=39999-step=240000.ckpt",
                                 data_module.basis_func_mtrx)
    source_rb = np.load(root / "prediced_coeff_source.npy")
    source_full_solution = np.load(root / "full_solution_source.npy")
    mse_rb = np.mean((rb_coeff_np - source_rb)**2)
    mse_full = np.mean((full_solution - source_full_solution)**2)

    print(f'{mse_full}')
    print(f'{mse_rb}')
    
    fig, ax = plt.subplots()
    ax.plot(source_full_solution, label = "Source Solution")
    ax.plot(full_solution, label = "My Solution")
    ax.legend()
    ax.grid()
    plt.show()