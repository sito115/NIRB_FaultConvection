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
import logging
import optuna
sys.path.append(str(Path(__file__).parents[1]))
from src.pod import match_scaler
from src.offline_stage import NirbDataModule, NirbModule, ComputeR2OnTrainEnd
from src.utils import load_pint_data, read_config, setup_logger, find_snapshot_path, find_basis_functions


def main():
    seed_everything(42) 
    IS_RUN_OPTUNA = True
    K_BEST_TRIALS = 3
    N_DEVICES = 3
    N_EPOCHS = 150_000 
    
    ### config
    config = read_config()
    ROOT = config['data_folder']
    assert ROOT.exists(), f"Not found: {ROOT}"
    FIELD_NAME = config['field_name']
    PROJECTION = config['projection']
    PARAMETER_SPACE = config['parameter_space']
    control_mesh_suffix =  config['control_mesh_suffix']

    if IS_RUN_OPTUNA:
        sqlite_file = ROOT / f"optuna_db{FIELD_NAME}.sqlite3"
        assert sqlite_file.exists()
        storage = f"sqlite:///{sqlite_file}"
        loaded_study = optuna.load_study(study_name="sweep", storage= storage)
        df_opt : pd.DataFrame = loaded_study.trials_dataframe()
        trials = [8, 21, 12, 6]
        df_opt_trunc = df_opt[df_opt["number"].isin(trials)]
    
    else:
        param = {
            'params_accuracy' : 1e-05 ,
            'params_batch_size' : 56,
            'params_lr': 1.83e-3,
            'params_scaler_output': 'min_max_init_grad', ##'min_max_init_grad',
            'params_scaler_features': 'standard', ##'min_max_init_grad',
            'params_activation': 'sigmoid',
            'layers' : [47, 198, 120, 96, 28]

        }
        
        df_opt_trunc = pd.DataFrame([param])
        
        # batch_size: 93
        # hidden_units:
        # - 79
        # - 92
        # - 66
        # learning_rate: 0.00022858237228051074
        # n_inputs: 1
        # n_outputs: 65
        # scaler_features: <src.pod.normalizer.MinMaxNormalizer object at 0x17d8611c0>
        # scaler_outputs: <src.pod.normalizer.MeanNormalizer object at 0x17d860830>

    for idx, row in df_opt_trunc.iterrows():
        logging.info(row)
        ACCURACY = row.params_accuracy if 'params_accuracy' in row.index else 1e-05
        BATCH_SIZE = row.params_batch_size 
        LR = row.params_lr
        SUFFIX = row.params_scaler_output #"min_max_init_grad"
        activation_name = row.params_activation
    
        if IS_RUN_OPTUNA:
            layers = [int(row.params_hidden1)]
            for i in range(row.params_num_inbetw_layers):
                layers.append(int(row[f'params_hidden_layers_betw{i}']))
            layers.append(int(row.params_hidden6))
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
    
        train_param_path = ROOT / "training_samples.csv"
        test_param_path = ROOT / "test_samples.csv"

        training_snapshots = np.load(find_snapshot_path(PROJECTION, SUFFIX, FIELD_NAME, ROOT, control_mesh_suffix, "Training"))[:, -1, :]
        test_snapshots     = np.load(find_snapshot_path(PROJECTION, SUFFIX, FIELD_NAME, ROOT, control_mesh_suffix, "Test"))[:, -1, :]
        
        basis_functions         = find_basis_functions(PROJECTION, SUFFIX, ACCURACY, FIELD_NAME, ROOT, control_mesh_suffix)
        training_parameters     = load_pint_data(train_param_path, is_numpy = True)
        test_parameters         = load_pint_data(test_param_path, is_numpy = True)
        
        if PARAMETER_SPACE == "01":
            training_parameters[:, 0] = np.log10(training_parameters[:, 0])
            test_parameters[:, 0] = np.log10(test_parameters[:, 0])


        if PARAMETER_SPACE == "05":
            training_snapshots = training_snapshots[:, np.newaxis, :]
            test_snapshots = test_snapshots[:, np.newaxis, :]
        
        assert len(training_snapshots) == len(training_parameters)

        scaling_output = match_scaler(SUFFIX)
        params_scaler_features = match_scaler(row.params_scaler_features)

        # random.seed(12342)
        # n_validation : int = 10
        # val_idx = random.sample(range(len(test_snapshots)), n_validation)  # upper bound is exclusive    
        # val_idx = np.arange(len(test_snapshots) - n_validation, len(test_snapshots))
        
        data_module = NirbDataModule(
            basis_func_mtrx=basis_functions,
            training_snaps=training_snapshots,
            training_param=training_parameters,
            test_param=test_parameters,
            test_snaps=test_snapshots,
            val_param=test_parameters,
            val_snaps=test_snapshots,
            batch_size=BATCH_SIZE,
            normalizer =scaling_output,
            standardizer_features=params_scaler_features
        )
        
        
        n_inputs = training_parameters.shape[1]
        n_outputs = basis_functions.shape[0]
        
        model = NirbModule(n_inputs,
                        layers,
                        n_outputs,
                        activation=activation_fn,
                        learning_rate=LR,
                        batch_size=BATCH_SIZE,
                        scaler_features=str(params_scaler_features),
                        scaler_outputs =str(scaling_output),
                        )
        

        r2_callback = ComputeR2OnTrainEnd(data_module.training_param_scaled,
                                        data_module.training_snaps_scaled,
                                        data_module.basis_func_mtrx)

        log_kwargs = {}
        log_kwargs['on_epoch'] = True
        if N_DEVICES > 1:
            log_kwargs['sync_dist'] = False
        
        model.log_kwargs = log_kwargs

        logger_dir_name = f"nn_logs_{ACCURACY:.1e}{SUFFIX}"
        logger = TensorBoardLogger(ROOT / f"TensorBoardLogs{FIELD_NAME}", name=logger_dir_name)
        logging.info(f'{logger.log_dir=}')
        
        model_ckpt = ModelCheckpoint(
            monitor = "val_loss",
            save_last=True,
            save_top_k=K_BEST_TRIALS,
            mode="min",
            filename = "{epoch:02d}-{step:02d}-{val_loss:.2f}",
            every_n_train_steps = 500
        )
        
        trainer = L.Trainer(max_epochs=N_EPOCHS,
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
        
        # model.basis_functions = data_module.basis_func_mtrx
        # model.val_snaps_scaled = data_module.val_snaps_scaled
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