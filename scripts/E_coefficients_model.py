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
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: F401
# from lightning.pytorch import seed_everything
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import random
import logging
import optuna
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbDataModule, NirbModule, ComputeR2OnTrainEnd, Normalizations
from src.utils import load_pint_data, plot_data, setup_logger


def main():
    # seed_everything(42) 
    IS_RUN_OPTUNA = True
    
    if IS_RUN_OPTUNA:
        sqlite_file = Path("/Users/thomassimader/Documents/NIRB/data/01/db_total_new.sqlite3")
        assert sqlite_file.exists()
        storage = f"sqlite:///{sqlite_file}"
        loaded_study = optuna.load_study(study_name="sweep1", storage= storage)
        df_opt : pd.DataFrame = loaded_study.trials_dataframe()
        trials = [87, 7, 38, 93, 74]
        df_opt_trunc = df_opt[df_opt["number"].isin(trials)]
    
    else:
        param = {
            'params_accuracy' : 1e-05 ,
            'params_batch_size' : 320,
            'params_lr': 1e-04,
            'params_normalization': 'mean_init_grad',
            'params_activation': 'sigmoid',
        }
        
        layers = [20, 250, 250, 100, 100]
        
        df_opt_trunc = pd.DataFrame(param)
    
    for idx, row in df_opt_trunc.iterrows():
        logging.info(row)
        ACCURACY = row.params_accuracy if 'params_accuracy' in row.index else 1e-05
        BATCH_SIZE = row.params_batch_size 
        LR = row.params_lr
        N_STEPS = 250_000 
        PARAMETER_SPACE = "01"
        ROOT = Path(__file__).parent.parent / "data" / PARAMETER_SPACE
        assert ROOT.exists(), f"Not found: {ROOT}"
        SUFFIX = row.params_normalization #"min_max_init_grad"
        N_DEVICES = 3
        activation_name = row.params_activation
        
        if IS_RUN_OPTUNA:
            layers = [int(row.params_hiden1)]
            for i in range(row.params_num_inbetw_layers):
                layers.append(int(row[f'params_hidden_layers_betw{i}']))
            layers.append(int(row.params_hiden6))
        else:
            layers = row.layers
    
        if activation_name == "leaky_relu":
            activation_fn = nn.LeakyReLU()
        elif activation_name == "relu":
            activation_fn = nn.ReLU()
        elif activation_name == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation_name =="tanh":
            activation_fn = nn.Tanh()
    
        control_mesh_suffix =  "s100_100_100_b0_4000_0_5000_-4000_0"
        
        if control_mesh_suffix is None:
            basis_func_path = ROOT / "TrainingMapped" /  "BasisFunctions" / f"basis_fts_matrix_{ACCURACY:.1e}{SUFFIX}.npy"
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                train_snapshots_path = ROOT / "TrainingMapped" / "Training_temperatures_minus_tgrad.npy" 
                test_snapshots_path = ROOT / "TestMapped" / "Test_temperatures_minus_tgrad.npy" 
            else:
                train_snapshots_path = ROOT / "TrainingMapped" /  "Training_temperatures.npy"
                test_snapshots_path = ROOT / "TestMapped" /  "Test_temperatures.npy" 
        else:
            basis_func_path = ROOT / "TrainingMapped" / control_mesh_suffix / "BasisFunctions" / f"basis_fts_matrix_{ACCURACY:.1e}{SUFFIX}.npy"
            if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                train_snapshots_path = ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / "Training_temperatures_minus_tgrad.npy" 
                test_snapshots_path = ROOT / "TestMapped" / control_mesh_suffix / "Exports" / "Test_temperatures_minus_tgrad.npy" 
            else:
                train_snapshots_path = ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / "Training_temperatures.npy"
                test_snapshots_path = ROOT / "TestMapped" / control_mesh_suffix / "Exports" / "Test_temperatures.npy" 
        train_param_path = ROOT / "training_samples.csv"
        test_param_path = ROOT / "test_samples.csv"

        
        basis_functions         = np.load(basis_func_path)
        training_snapshots      = np.load(train_snapshots_path)
        training_parameters     = load_pint_data(train_param_path, is_numpy = True)
        test_snapshots          = np.load(test_snapshots_path)
        test_parameters         = load_pint_data(test_param_path, is_numpy = True)
        

        
        if PARAMETER_SPACE == "01":
            training_parameters[:, 0] = np.log10(training_parameters[:, 0])
            test_parameters[:, 0] = np.log10(test_parameters[:, 0])


        if PARAMETER_SPACE == "05":
            training_snapshots = training_snapshots[:, np.newaxis, :]
            test_snapshots = test_snapshots[:, np.newaxis, :]
            

        # mask = ~(training_snapshots == 0).all(axis=(1, 2)) # omit indices that are not computed yet
        # training_snapshots       = training_snapshots[mask]
        # training_parameters      = training_parameters[mask, :]
        
        assert len(training_snapshots) == len(training_parameters)
        # Prepare data

        if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
            training_snapshots  = training_snapshots[:, -1, :] # last time step
            test_snapshots      = test_snapshots[:, -1, :]
            logging.debug("Entered 'init' and 'grad' condition")
        elif 'init' in SUFFIX.lower():
            training_snapshots  = training_snapshots[:, -1, :] -  training_snapshots[:, 0, :] 
            test_snapshots      = test_snapshots[:, -1, :] - test_snapshots[:, 0, :]
            logging.debug("Entered 'init' condition")
        else:
            training_snapshots  = training_snapshots[:, -1, :]
            test_snapshots      = test_snapshots[:, -1, :]
            logging.debug("Entered else statement condition")
        

        if "mean" in SUFFIX.lower():
            scaling = Normalizations.Mean
        elif "min_max" in SUFFIX.lower():
            scaling = Normalizations.MinMax
        else:
            raise ValueError("Invalid suffix.")
        print(f"Selected {scaling}")

        random.seed(12342)
        n_validation : int = 10
        # val_idx = random.sample(range(len(test_snapshots)), n_validation)  # upper bound is exclusive    
        val_idx = np.arange(len(test_snapshots) - n_validation, len(test_snapshots))
        data_module = NirbDataModule(
            basis_func_mtrx=basis_functions,
            training_snaps=training_snapshots,
            training_param=training_parameters,
            test_param=test_parameters,
            test_snaps=test_snapshots,
            val_param=test_parameters,
            val_snaps=test_snapshots,
            batch_size=BATCH_SIZE,
            normalizer =scaling,
        )
        
        
        def plot_scaled_data():
            plot_data(data_module.training_snaps,
                    title = "Training Snapshots - Unscaled",
                    export_path = ROOT / "Training - Unscaled.png")
            plot_data(data_module.test_snaps,
                    title = "Test Snapshots - Unscaled",
                    export_path = ROOT / "Test - Unscaled.png")
            plot_data(data_module.training_snaps_scaled,
                    title = "Training Snapshots - Scaled",
                    export_path = ROOT / f"Training - Scaled{SUFFIX}.png")
            plot_data(data_module.test_snaps_scaled,
                    title = "Test Snapshots - Scaled",
                    export_path = ROOT / f"Test - Scaled{SUFFIX}.png")
        plot_scaled_data()
        
        n_inputs = training_parameters.shape[1]
        n_outputs = basis_functions.shape[0]
        
        model = NirbModule(n_inputs,
                        layers,
                        n_outputs,
                        activation=activation_fn,
                        learning_rate=LR,
                        batch_size=BATCH_SIZE)
        

        r2_callback = ComputeR2OnTrainEnd(data_module.training_param_scaled,
                                        data_module.training_snaps_scaled,
                                        data_module.basis_func_mtrx)

        log_kwargs = {}
        log_kwargs['on_epoch'] = True
        if N_DEVICES > 1:
            log_kwargs['sync_dist'] = False
        
        model.log_kwargs = log_kwargs

        logger_dir_name = f"nn_logs_{ACCURACY:.1e}{SUFFIX}"
        logger = TensorBoardLogger(ROOT, name=logger_dir_name)
        
        model_ckpt = ModelCheckpoint(
            monitor = "val_loss",
            save_last=True,
            save_top_k=10,
            mode="min",
            filename = "{epoch:02d}-{val_loss:.2f}",
            every_n_train_steps = 400
        )
        
        trainer = L.Trainer(max_steps=N_STEPS,
                            logger=logger,
                            callbacks=[r2_callback,model_ckpt], #, EarlyStopping("Q2_val", mode="min")],
                            strategy='ddp',
                            enable_progress_bar=False,
                            devices=N_DEVICES,
                            accelerator= "cpu", #'mps',
                            log_every_n_steps = BATCH_SIZE
                            )
        
        try:
            ckpt_folder = ROOT / logger_dir_name / "version_12" / "checkpoints"
            ckpt_path = [path for path in ckpt_folder.iterdir() if path.suffix == ".ckpt"][0]
            print(ckpt_path)
        except FileNotFoundError:
            pass
        
        model.basis_functions = data_module.basis_func_mtrx
        model.val_snaps_scaled = data_module.val_snaps_scaled
        trainer.fit(model=model,
                    train_dataloaders=data_module.train_dataloader(shuffle = True),
                    val_dataloaders=data_module.validation_dataloader(shuffle = False),
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
        model.basis_functions = data_module.basis_func_mtrx
        model.test_snaps_scaled = data_module.test_snaps_scaled
        results = test_trainer.test(model=model,
                                    dataloaders=data_module.test_dataloader())
        print(results)


if __name__ == "__main__":
    setup_logger(is_console=True, level=logging.INFO)
    main()