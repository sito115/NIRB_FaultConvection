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
from lightning.pytorch.callbacks import EarlyStopping  # noqa: F401
from lightning.pytorch import seed_everything
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbDataModule, NirbModule, ComputeR2OnTrainEnd, Normalizations
from src.utils import load_pint_data, plot_data


def main():
    seed_everything(42) 
    ACCURACY = 1e-5
    BATCH_SIZE = 50
    LR = 1e-4
    N_STEPS = 250_000 
    ROOT = Path(__file__).parent.parent / "data" / "01"
    assert ROOT.exists(), f"Not found: {ROOT}"
    SUFFIX = "mean"
    
    control_mesh_suffix =  "s100_100_100_b0_4000_0_5000_-4000_0"
    basis_func_path = ROOT / "TrainingMapped" / control_mesh_suffix / "BasisFunctions" / f"basis_fts_matrix_{ACCURACY:.1e}{SUFFIX}.npy"
    train_snapshots_path = ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / "Training_temperatures.npy"
    test_snapshots_path = ROOT / "TestMapped" / control_mesh_suffix / "Exports" / "Test_temperatures.npy"
    train_param_path = ROOT / "training_samples.csv"
    test_param_path = ROOT / "test_samples.csv"

    
    basis_functions         = np.load(basis_func_path)
    training_snapshots      = np.load(train_snapshots_path)
    training_parameters     = load_pint_data(train_param_path, is_numpy = True)
    test_snapshots          = np.load(test_snapshots_path)
    test_parameters         = load_pint_data(test_param_path, is_numpy = True)
    
    mask = ~(training_snapshots == 0).all(axis=(1, 2)) # omit indices that are not computed yet
    training_snapshots      = training_snapshots[mask]
    training_parameters      = training_parameters[mask, :]
    assert len(training_parameters) == len(training_parameters)
    # Prepare data
    step = 1
    training_snapshots = training_snapshots[::step, -1, :] # last time step
    training_parameters = training_parameters[::step, :]
    test_snapshots = test_snapshots[:, -1, :] # last time step
    
    if "mean" in SUFFIX.lower():
        scaling = Normalizations.Mean
    elif "min_max" in SUFFIX.lower():
        scaling = Normalizations.MinMax
    else:
        raise ValueError("Invalid suffix.")
    
    
    data_module = NirbDataModule(
        basis_func_mtrx=basis_functions,
        training_snaps=training_snapshots,
        training_param=training_parameters,
        test_param=test_parameters,
        test_snaps=test_snapshots,
        batch_size=BATCH_SIZE,
        normalizer =scaling,
    )
    
    # plot_data(data_module.training_snaps,
    #           title = "Training Snapshots - Unscaled",
    #           export_path = ROOT / "Training - Unscaled.png")
    # plot_data(data_module.test_snaps,
    #           title = "Test Snapshots - Unscaled",
    #           export_path = ROOT / "Test - Unscaled.png")
    # plot_data(data_module.training_snaps_scaled,
    #           title = "Training Snapshots - Scaled",
    #           export_path = ROOT / f"Training - Scaled{SUFFIX}.png")
    # plot_data(data_module.test_snaps_scaled,
    #           title = "Test Snapshots - Scaled",
    #           export_path = ROOT / f"Test - Scaled{SUFFIX}.png")
    
    n_inputs = training_parameters.shape[1]
    n_outputs = basis_functions.shape[0]
    
    model = NirbModule(n_inputs,
                    [100, 250, 100]  ,
                    n_outputs,
                    activation=nn.Sigmoid(),
                    learning_rate=LR,
                    batch_size=BATCH_SIZE)
    
    # summary(model.model, 
    #         input_size=training_parameters.shape,
    #         col_names=["input_size",
    #                    "output_size",
    #                    "num_params"],)

    r2_callback = ComputeR2OnTrainEnd(data_module.training_param_scaled,
                                    data_module.training_snaps_scaled,
                                    data_module.basis_func_mtrx)

    logger_dir_name = f"nn_logs_{ACCURACY:.1e}{SUFFIX}"
    logger = TensorBoardLogger(ROOT, name=logger_dir_name)
    trainer = L.Trainer(max_steps=N_STEPS,
                        logger=logger,
                        log_every_n_steps=BATCH_SIZE*10,  # Reduce logging frequency
                        callbacks=[r2_callback], #, EarlyStopping("Q2_val", mode="min")],
                        strategy='ddp',
                        enable_progress_bar=False,
                        profiler="simple",
                        devices=1,
                        accelerator= "cpu", #'mps',
                        )
    
    try:
        ckpt_folder = ROOT / logger_dir_name / "version_18" / "checkpoints"
        ckpt_path = [path for path in ckpt_folder.iterdir() if path.suffix == ".ckpt"][0]
        print(ckpt_path)
    except FileNotFoundError:
        pass
    
    val_ind = np.arange(20)
    model.val_snaps_scaled = data_module.test_snaps_scaled[val_ind]
    model.basis_functions = basis_functions
    trainer.fit(model=model,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.validation_dataloader(val_ind),
                ckpt_path = None) #ckpt_path
    
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


if __name__ == "__main__":
    main()