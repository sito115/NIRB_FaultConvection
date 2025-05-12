"""
Define a neural network (NN) to predict the coefficient for each basis function from "D_pod.py".
The input are the respective parameters from A_sampling.py 
The training and test outputs (coefficients for basis functions) are computed in the NirbDataModule class.
Everything for the definition of the NN is defined in the NirbModule(L.LightningModule) class.
"""

import lightning as L
from torchinfo import summary # noqa: F401
from torch import nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from scr.offline_stage import NirbDataModule, NirbModule, ComputeR2OnTrainEnd, Normalizations
from scr.utils import load_pint_data, plot_data

if __name__ == "__main__":
    seed_everything(42) 
    ACCURACY = 1e-6
    BATCH_SIZE = 50
    LR = 8e-4 #0.00141 # 0.008656381123933186 # Trial 93
    N_EPOCHS = 400_000 
    ROOT = Path(__file__).parent.parent / "data" / "01"
    assert ROOT.exists(), f"Not found: {ROOT}"
    
    # Best trial: for 1e-4
    #   Value:  0.030139805749058723
    #   Params: 
    #     hiden1: 24
    #     num_inbetw_layers: 3
    #     hidden_layers_betw0: 193
    #     hidden_layers_betw1: 193
    #     hidden_layers_betw2: 101
    #     hiden6: 96
    #     lr: 0.0015856631398601722
    #     batch_size: 15
    
    
    basis_func_path = ROOT / "BasisFunctions" / f"basis_fts_matrix_{ACCURACY:.1e}min_max.npy"
    train_snapshots_path = ROOT / "TrainingMapped" / "Training_temperatures.npy"
    test_snapshots_path = ROOT / "TestMapped" / "Test_temperatures.npy"
    train_param_path = ROOT / "training_samples.csv"
    test_param_path = ROOT / "test_samples.csv"
    
    basis_functions         = np.load(basis_func_path)
    training_snapshots      = np.load(train_snapshots_path)
    training_parameters     = load_pint_data(train_param_path, is_numpy = True)
    test_snapshots          = np.load(test_snapshots_path)
    test_parameters         = load_pint_data(test_param_path, is_numpy = True)
    
    # Prepare data
    training_snapshots = training_snapshots[:, -1, :] # last time step
    test_snapshots = test_snapshots[:, -1, :] # last time step
     
    data_module = NirbDataModule(
        basis_func_mtrx=basis_functions,
        training_snaps=training_snapshots,
        training_param=training_parameters,
        test_param=test_parameters,
        test_snaps=test_snapshots,
        batch_size=BATCH_SIZE,
        normalizer = Normalizations.MinMax
    )
    
    plot_data(data_module.training_snaps,
              title = "Training Snapshots - Unscaled",
              export_path = ROOT / "Training - Unscaled.png")
    plot_data(data_module.training_snaps_scaled,
              title = "Training Snapshots - Scaled",
              export_path = ROOT / "Training - Scaled.png")
    plot_data(data_module.test_snaps_scaled,
              title = "Test Snapshots - Scaled",
              export_path = ROOT / "Test - Scaled.png")
    
    n_inputs = training_parameters.shape[1]
    n_outputs = basis_functions.shape[0]
    
    model = NirbModule(n_inputs,
                       [28, 393, 28] ,
                       n_outputs,
                       activation=nn.Sigmoid(),
                       learning_rate=LR)
    
    # summary(model.model, 
    #         input_size=training_parameters.shape,
    #         col_names=["input_size",
    #                    "output_size",
    #                    "num_params"],)

    r2_callback = ComputeR2OnTrainEnd(data_module.training_param_scaled,
                                      data_module.training_snaps_scaled,
                                      data_module.basis_func_mtrx)

    logger_dir_name = f"nn_logs_{ACCURACY:.1e}"
    logger = TensorBoardLogger(ROOT, name=logger_dir_name)
    trainer = L.Trainer(max_epochs=N_EPOCHS,
                        logger=logger,
                        log_every_n_steps=BATCH_SIZE*10,  # Reduce logging frequency
                        callbacks=[r2_callback],
                        strategy='ddp',
                        enable_progress_bar=False,
                        profiler="simple",
                        devices=4,
                        accelerator= "cpu" #'mps',
                        )
    
    try:
        ckpt_folder = ROOT / logger_dir_name / "version_3" / "checkpoints"
        ckpt_path = [path for path in ckpt_folder.iterdir() if path.suffix == ".ckpt"][0]
        print(ckpt_path)
    except FileNotFoundError:
        pass
    
    trainer.fit(model=model,
                train_dataloaders=data_module.train_dataloader(shuffle = False),
                ckpt_path = ckpt_path) #ckpt_path
    
    del trainer
    del model
    
    test_trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            logger=logger,  # <-- same logger
            )
    
    model = NirbModule.load_from_checkpoint([path for path in Path(logger.log_dir).rglob("*.ckpt")][0])
    model.eval()
    model.test_snaps_scaled = data_module.test_snaps_scaled
    model.basis_functions = basis_functions
    results = test_trainer.test(model=model,
                                dataloaders=data_module.test_dataloader())
    print(results)