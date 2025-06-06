import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import optuna
from torch import nn
import numpy as np
import sys 
import logging
sys.path.append(str(Path(__file__).parents[1]))
from src.offline_stage import NirbModule, NirbDataModule, ComputeR2OnTrainEnd, OptunaPruning, Normalizations
from src.utils import load_pint_data

def objective(trial: optuna.Trial) -> float:
    # Architecture: variable number of layers and neurons per layer
    hidden1 = trial.suggest_int("hiden1", low = 2, high = 300)
    num_inbetw_layers = trial.suggest_int("num_inbetw_layers", 1, 4)
    hidden_layers_betw = [
        trial.suggest_int(f"hidden_layers_betw{i}", 50, 400, step=2)
        for i in range(num_inbetw_layers)
    ]
    hidden6 = trial.suggest_int("hiden6", low = 30,  high = 300 ,step=2)
    
    suffix = trial.suggest_categorical("scaler_output", ["min_max_init_grad", "mean"])
    scaler_features = trial.suggest_categorical("scaler_features", ["Standardizer", "Mean", "MinMax"])
    accuracy = trial.suggest_categorical("accuracy", [1e-5, 1e-6])
    
    # Other hyperparameters
    lr = trial.suggest_float("lr", 5e-6, 2e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 10, 100)

    activation_name = trial.suggest_categorical("activation", ["sigmoid", "relu", "leaky_relu", "tanh"])
    if activation_name == "leaky_relu":
        activation_fn = nn.LeakyReLU()
    elif activation_name == "relu":
        activation_fn = nn.ReLU()
    elif activation_name == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation_name =="tanh":
        activation_fn = nn.Tanh()

    

    if PROJECTION == "Mapped":
        export_root_train = ROOT / f"Training{PROJECTION}" / control_mesh_suffix / "Exports"
        export_root_test = ROOT / f"Test{PROJECTION}" / control_mesh_suffix / "Exports"
        basis_func_path = ROOT / "TrainingMapped" / control_mesh_suffix / "BasisFunctions" / f"basis_fts_matrix_{accuracy:.1e}{suffix}.npy"

    else:
        export_root_train = ROOT / f"Training{PROJECTION}" 
        export_root_test = ROOT / f"Test{PROJECTION}"
        basis_func_path = ROOT / "BasisFunctions" / f"basis_fts_matrix_{accuracy:.1e}{suffix}.npy"


    assert export_root_train.exists(), f"Export root train {export_root_train} does not exist."
    assert export_root_test.exists(), f"Export root test {export_root_test} does not exist."
    assert basis_func_path.exists(), f"Basis function path {basis_func_path} does not exist."
    basis_functions         = np.load(basis_func_path)
    
    if 'init' in suffix.lower() and 'grad' in suffix.lower():
        logging.debug("Entered 'init' and 'grad' condition")
        training_snapshots_npy      = np.load(export_root_train / "Training_Temperature_minus_tgrad.npy")
        test_snapshots_npy          = np.load(export_root_test / "Test_Temperature_minus_tgrad.npy")
    else:
        logging.debug("Entered else statement condition")
        training_snapshots_npy      = np.load(export_root_train / "Training_Temperature.npy")
        test_snapshots_npy          = np.load(export_root_test / "Test_Temperature.npy")
    training_snapshots  = training_snapshots_npy[:, -1, :]
    test_snapshots      = test_snapshots_npy[:, -1, :]
        
        
    train_param_path = ROOT / "training_samples.csv"
    test_param_path = ROOT / "test_samples.csv"
    
    
    training_parameters     = load_pint_data(train_param_path, is_numpy = True)
    test_parameters         = load_pint_data(test_param_path, is_numpy = True)
    
    # mask = ~(training_snapshots == 0).all(axis=(1, 2)) # omit indices that are not computed yet
    # training_snapshots       = training_snapshots[mask]
    # training_parameters      = training_parameters[mask, :]
    
    
    if PARAMETER_SPACE == "01":
        training_parameters[:, 0] = np.log10(training_parameters[:, 0])
        test_parameters[:, 0] = np.log10(test_parameters[:, 0])
    
    assert len(training_snapshots) == len(training_parameters)
    # Prepare data

    if scaler_features.lower() == "standardizer":
        scaler_features = Normalizations.Standardizer
    elif scaler_features.lower() == "mean":
        scaler_features = Normalizations.Mean
    elif scaler_features.lower() == "min_max":
        scaler_features = Normalizations.MinMax
    else:
        raise ValueError("Invalid scaler for features.")


    if "mean" in suffix.lower():
        scaling_outputs = Normalizations.Mean
    elif "min_max" in suffix.lower():
        scaling_outputs = Normalizations.MinMax
    else:
        raise ValueError("Invalid suffix.")

    data_module = NirbDataModule(
        basis_func_mtrx=basis_functions,
        training_snaps=training_snapshots,
        training_param=training_parameters,
        val_param=test_parameters,
        val_snaps=test_snapshots,
        batch_size=batch_size,
        standardizer_features=scaler_features,
        normalizer =scaling_outputs,
    )
    
    n_inputs = training_parameters.shape[1]
    n_outputs = basis_functions.shape[0]
    
    model = NirbModule(n_inputs, 
                       [hidden1] + hidden_layers_betw + [hidden6],
                       n_outputs,
                       activation=activation_fn,
                       learning_rate=lr,
                       batch_size=batch_size,
                       scaler_features=scaler_features,
                       scaler_outputs=scaling_outputs,)

    optuna_pruning = OptunaPruning(
        trial, 
        monitor="val_loss",        # or "val_loss"
        check_val_every_n_steps=20,
        mode="min",                  # we're minimizing loss
        )
    
    
    r2_callback = ComputeR2OnTrainEnd(data_module.training_param_scaled,
                                      data_module.training_snaps_scaled,
                                      data_module.basis_func_mtrx)

    model_ckpt = ModelCheckpoint(
            monitor = "Q2_val",
            save_last=True,
            save_top_k=3,
            mode="min",
            filename = "{epoch:02d}-{Q2_val:.2e}",
            every_n_train_steps = 500
        )

    logger_dir_name = "optuna_logs"
    logger = TensorBoardLogger(ROOT, name=logger_dir_name,
                               version=f"trial_{trial.number}",
                               default_hp_metric=False)
    trainer = L.Trainer(max_steps=N_STEPS,
                        enable_checkpointing=True,
                        logger=logger,
                        callbacks=[r2_callback, optuna_pruning, model_ckpt],
                        enable_progress_bar=False,
                        max_time={"minutes": 70},
                        )
    
    model.basis_functions = data_module.basis_func_mtrx
    model.val_snaps_scaled = data_module.val_snaps_scaled
    trainer.fit(model,
                train_dataloaders=data_module.train_dataloader(shuffle = True,
                                                                num_workers = N_JOBS,
                                                               persistent_workers=True),
                val_dataloaders=data_module.validation_dataloader(num_workers = 1,
                                                                  persistent_workers=True)
                )
    
    # results = trainer.test(model, dataloaders=data_module.test_dataloader())
    # test_loss = results[0]['test_loss']
    train_loss = model.train_loss
    trial.set_user_attr("train_loss", float(train_loss))
    # trial.set_user_attr("test_loss", test_loss)
    val_loss = trainer.callback_metrics.get("val_loss")
    return float(val_loss)

if __name__ == "__main__":
    N_STEPS = 100_000 #100_000 #20_000
    PROJECTION = "Original"  # "Original" or "Mapped"
    control_mesh_suffix =  "s100_100_100_b0_4000_0_5000_-4000_-0"
    # ACCURACY = 1e-5
    # SUFFIX = "min_max_init_grad"
    PARAMETER_SPACE = "07"
    ROOT = Path(__file__).parents[1] / "data" / PARAMETER_SPACE
    STUDY_NAME = "sweep"
    N_JOBS = 4
    N_TRIALS = 150


    assert ROOT.exists(), f"Not found: {ROOT}"
    db_path = ROOT / "optuna_db.sqlite3" #/f"db_{ACCURACY:.1e}{SUFFIX}.sqlite3"
    storage_param = {
        "storage": f"sqlite:///{db_path}",  # Specify the storage URL here.
        "study_name": STUDY_NAME,
        "load_if_exists": True
    }
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=N_TRIALS * 0.33,
                                    n_warmup_steps=int(N_STEPS*0.33),
                                    n_min_trials=20
                                ),
                                **storage_param)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)