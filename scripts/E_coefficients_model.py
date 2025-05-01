import lightning as L
from torchinfo import summary
from torch import nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from scr.offline_stage import NirbDataModule, NirbModule, ComputeR2OnTrainEnd
from scr.utils import load_pint_data

if __name__ == "__main__":
    seed_everything(42) 
    ACCURACY = 1e-3
    BATCH_SIZE = 13
    LR = 0.00141 # 0.008656381123933186 # Trial 93
    N_EPOCHS = 50_000
    ROOT = Path(__file__).parent.parent / "data" / "01"
    assert ROOT.exists(), f"Not found: {ROOT}"
    
    basis_func_path = ROOT / "BasisFunctions" / f"basis_fts_matrix_{ACCURACY:.1e}.npy"
    train_snapshots_path = ROOT / "Exports" / "Training_temperatures.npy"
    test_snapshots_path = ROOT / "Test" / "Test_temperatures.npy"
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
        test_param=training_parameters,
        test_snaps=training_snapshots,
        batch_size=20,
    )
    
    n_inputs = training_parameters.shape[1]
    n_outputs = basis_functions.shape[0]
    
    model = NirbModule(n_inputs,
                       [13, 260, 275, 295, 41],
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

    logger = TensorBoardLogger(ROOT, name="nn_logs")
    trainer = L.Trainer(max_epochs=N_EPOCHS,
                        logger=logger,
                        log_every_n_steps=BATCH_SIZE*10,  # Reduce logging frequency
                        callbacks=[r2_callback],
                        strategy='ddp',
                        enable_progress_bar=False,
                        profiler="simple",
                        devices=5,
                        accelerator= "cpu" #'mps',
                        )
    

    # ckpt_path = "/Users/thomassimader/Documents/NIRB/Snapshots/01/nn_logs/version_41/checkpoints/epoch=199999-step=200000.ckpt"
    trainer.fit(model=model,
                train_dataloaders=data_module.train_dataloader(),
                ckpt_path = None)
    model.test_snaps_scaled = data_module.test_snaps_scaled
    results = trainer.test(model=model,
                 dataloaders=data_module.test_dataloader())
    print(results)