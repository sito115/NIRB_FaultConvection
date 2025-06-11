"""
Define a neural network (NN) to predict the coefficient for each basis function from "D_pod.py".
The input are the respective parameters from A_sampling.py 
The training and test outputs (coefficients for basis functions) are computed in the NirbDataModule class.
Everything for the definition of the NN is defined in the NirbModule(L.LightningModule) class.
"""
import lightning as L
from torch import nn
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: F401
from lightning.pytorch import seed_everything
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import random
import logging
import optuna
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbDataModule, NirbModule, ComputeR2OnTrainEnd
from src.utils import load_pint_data, plot_data, setup_logger
from src.pod.normalizer import MeanNormalizer, MinMaxNormalizer, Standardizer


def main():
    seed_everything(42) 
    IS_RUN_OPTUNA = False
    
    
    N_STEPS = 50_000 
    PARAMETER_SPACE = "01"
    ROOT = Path(__file__).parent.parent / "data" / PARAMETER_SPACE
    FIELD_NAME = "EntropyNum"
    PROJECTION = "Mapped"
    assert ROOT.exists(), f"Not found: {ROOT}"
    N_DEVICES = 2

    
    if IS_RUN_OPTUNA:
        sqlite_file = ROOT / "optuna_db.sqlite3"
        assert sqlite_file.exists()
        storage = f"sqlite:///{sqlite_file}"
        loaded_study = optuna.load_study(study_name="sweep", storage= storage)
        df_opt : pd.DataFrame = loaded_study.trials_dataframe()
        trials = [148, 67, 114, 100, 71, 81, 110, 145]
        df_opt_trunc = df_opt[df_opt["number"].isin(trials)]
    
    else:
        param = {
            'params_accuracy' : 1e-05 ,
            'params_batch_size' : 10,
            'params_lr': 1e-3,
            'params_scaler_output': 'min_max', ##'min_max_init_grad',
            'params_activation': 'sigmoid',
            'params_scaler_features': 'standard',
            'layers' : [15, 15, 15]

        }
        df_opt_trunc = pd.DataFrame([param])
        
    for idx, row in df_opt_trunc.iterrows():
        logging.info(row)
        ACCURACY = row.params_accuracy if 'params_accuracy' in row.index else 1e-05
        BATCH_SIZE = row.params_batch_size 
        LR = row.params_lr
        SUFFIX = row.params_scaler_output #"min_max_init_grad"
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
    
        
        
        if PARAMETER_SPACE == "07":
            match FIELD_NAME:
                case "Temperature":
                    if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                        train_snapshots_path = ROOT / "TrainingOriginal" / f"Training_{FIELD_NAME}_minus_tgrad.npy" 
                        test_snapshots_path = ROOT / "TestOriginal" / f"Test_{FIELD_NAME}_minus_tgrad.npy" 
                    else:
                        train_snapshots_path = ROOT / "TrainingOriginal" /  f"Training_{FIELD_NAME}.npy"
                        test_snapshots_path = ROOT / "TestOriginal" /  f"Test_{FIELD_NAME}.npy" 
                case "EntropyNum":
                    train_snapshots_path =  ROOT / "TrainingOriginal" / "Training_entropy_gen_number_therm.npy"
                    test_snapshots_path =  ROOT / "TestOriginal" / "Test_entropy_gen_number_therm.npy"
            basis_func_folder = ROOT / f"BasisFunctionsPerZeroCrossing{FIELD_NAME}"
                
        elif PARAMETER_SPACE == "01":
            control_mesh_suffix =  "s100_100_100_b0_4000_0_5000_-4000_0"
            match FIELD_NAME:
                case "Temperature":
                    if 'init' in SUFFIX.lower() and 'grad' in SUFFIX.lower():
                        train_snapshots_path = ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / f"Training_{FIELD_NAME}_minus_tgrad.npy" 
                        test_snapshots_path = ROOT / "TestMapped" / control_mesh_suffix / "Exports" / f"Test_{FIELD_NAME}_minus_tgrad.npy" 
                    else:
                        train_snapshots_path = ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / f"Training_{FIELD_NAME}.npy"
                        test_snapshots_path = ROOT / "TestMapped" / control_mesh_suffix / "Exports" / f"Test_{FIELD_NAME}.npy" 
                case "EntropyNum":
                    train_snapshots_path =  ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" /"entropy_gen_number.npy"
                    test_snapshots_path =  ROOT / "TestMapped" / control_mesh_suffix / "Exports" / "Test_entropy_gen_number.npy"
            
            basis_func_folder = ROOT / "TrainingMapped" / control_mesh_suffix / f"BasisFunctionsPerZeroCrossing{FIELD_NAME}"
            
        else:
            raise NotImplementedError(f"Paths for parameter space {PARAMETER_SPACE} not implemented yet.")
        
        
        assert basis_func_folder.exists(), f"Basis function folder {basis_func_folder} does not exist."

        train_param_path = ROOT / "training_samples.csv"
        test_param_path = ROOT / "test_samples.csv"

        

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
        
        assert len(training_snapshots) == len(training_parameters)

        match FIELD_NAME:
            case "Temperature":
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
            case "EntropyNum":
                training_snapshots = training_snapshots[:, -1:]
                test_snapshots  = test_snapshots[:, -1:]
                

        if "mean" in SUFFIX.lower():
            scaling_output = MeanNormalizer()
        elif "min_max" in SUFFIX.lower():
            scaling_output = MinMaxNormalizer()
        elif "standard" in SUFFIX.lower():  
            scaling_output = Standardizer()
        else:
            raise ValueError("Invalid suffix.")
        print(f"Selected {scaling_output}")

        training_snapshots_scaled = scaling_output.normalize(training_snapshots)
        test_snapshots_scaled = scaling_output.normalize_reuse_param(test_snapshots)


        params_scaler_features_str : str = row.params_scaler_features
        if "mean" in params_scaler_features_str.lower():
            params_scaler_features = MeanNormalizer()
        elif "min_max" in params_scaler_features_str.lower():
            params_scaler_features = MinMaxNormalizer()
        elif "standard" in params_scaler_features_str.lower():
            params_scaler_features = Standardizer()
        else:
            raise ValueError("Invalid suffix.")
        print(f"Selected {params_scaler_features}") 

        training_parameters_scaled = params_scaler_features.normalize(training_parameters)
        test_parameters_scaled = params_scaler_features.normalize_reuse_param(training_parameters)


        def plot_scaled_data():
            plot_data(training_snapshots,
                    title = "Training Snapshots - Unscaled",
                    export_path = ROOT / "Training - Unscaled.png")
            plot_data(test_snapshots,
                    title = "Test Snapshots - Unscaled",
                    export_path = ROOT / "Test - Unscaled.png")
            plot_data(training_snapshots_scaled,
                    title = "Training Snapshots - Scaled",
                    export_path = ROOT / f"Training - Scaled{SUFFIX}.png")
            plot_data(test_snapshots_scaled,
                    title = "Test Snapshots - Scaled",
                    export_path = ROOT / f"Test - Scaled{SUFFIX}.png")
        # plot_scaled_data()

        random.seed(12342)
        
        zero_crossings_train = np.load(ROOT / "Exports" / "Training_zero_crossings.npy")
        zero_crossings_test = np.load(ROOT / "Exports" / "Test_zero_crossings.npy")
        assert len(zero_crossings_train) == len(training_snapshots)
        assert len(zero_crossings_test) == len(test_snapshots)
        unique_zc = np.unique(zero_crossings_train)
        grouped_zc_train = {int(val): np.where(zero_crossings_train == val)[0] for val in unique_zc }
        grouped_zc_test = {int(val): np.where(zero_crossings_test == val)[0] for val in unique_zc}


        basis_func_paths = sorted([path for path in basis_func_folder.rglob(f"basis_fts_matrix_{FIELD_NAME}_{ACCURACY:.1e}{SUFFIX}_zc*.npy")])
        # - 1 because we do not have a basis function for the zero crossing 0, it is inclued in all others.
        assert len(basis_func_paths) == len(unique_zc) - 1, f"Number of basis function files {len(basis_func_paths)} does not match number of unique zero crossings {len(unique_zc)}."
        
        for basis_func_path in basis_func_paths:
            basis_functions = np.load(basis_func_path)
            zc = int(basis_func_path.stem.split("_")[-1][2:])
            logging.info(f"Processing zero crossing {zc} with basis function path {basis_func_path}.")
            assert zc in unique_zc, f"Zero crossing {zc} not found in unique zero crossings."
            indices_train = grouped_zc_train.get(zc, [])
            indices_with_0_zc_train = np.append(indices_train, grouped_zc_train.get(0, []))
            indices_test = grouped_zc_test.get(zc, [])
            indices_with_0_zc_test = np.append(indices_test, grouped_zc_test.get(0, []))
            
            data_module = NirbDataModule(
                basis_func_mtrx=basis_functions,
                training_snaps=training_snapshots_scaled[indices_with_0_zc_train],
                training_param=training_parameters_scaled[indices_with_0_zc_train],
                test_param=test_parameters_scaled[indices_with_0_zc_test],
                test_snaps=test_snapshots_scaled[indices_with_0_zc_test],
                val_param=test_parameters_scaled[indices_with_0_zc_test],
                val_snaps=test_snapshots_scaled[indices_with_0_zc_test],
                batch_size=BATCH_SIZE,
                normalizer =None,
                standardizer_features=None,
            )
        
    
            n_inputs = training_parameters.shape[1]
            n_outputs = basis_functions.shape[0]
        
            model = NirbModule(n_inputs,
                            layers,
                            n_outputs,
                            activation=activation_fn,
                            learning_rate=LR,
                            batch_size=BATCH_SIZE,
                            standardizer_features=str(params_scaler_features),
                            normalizer =str(scaling_output),
                            )
        

            r2_callback = ComputeR2OnTrainEnd(data_module.training_param_scaled,
                                            data_module.training_snaps_scaled,
                                            data_module.basis_func_mtrx)

            log_kwargs = {}
            log_kwargs['on_epoch'] = True
            if N_DEVICES > 1:
                log_kwargs['sync_dist'] = False
            
            model.log_kwargs = log_kwargs

            logger_dir_name = f"nn_logs_{ACCURACY:.1e}{SUFFIX}{FIELD_NAME}_zc{zc}"
            logger = TensorBoardLogger(ROOT, name=logger_dir_name)
            logging.info(f'{logger.log_dir=}')
            
            model_ckpt = ModelCheckpoint(
                monitor = "val_loss",
                save_last=True,
                save_top_k=3,
                mode="min",
                filename = "{epoch:02d}-{step:02d}-{val_loss:.2f}",
                every_n_train_steps = 300
            )
            
            trainer = L.Trainer(max_steps=N_STEPS,
                                logger=logger,
                                callbacks=[r2_callback,model_ckpt],
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
            

            model.basis_functions = data_module.basis_func_mtrx
            model.test_snaps_scaled = data_module.test_snaps_scaled
            results = trainer.test(model=model,
                                        dataloaders=data_module.test_dataloader())
            print(results)


if __name__ == "__main__":
    setup_logger(is_console=True, level=logging.INFO)
    main()