import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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
    num_inbetw_layers = trial.suggest_int("num_inbetw_layers", 1, 5)
    hidden_layers_betw = [
        trial.suggest_int(f"hidden_layers_betw{i}", 50, 400, step=2)
        for i in range(num_inbetw_layers)
    ]
    hidden6 = trial.suggest_int("hiden6", low = 30,  high = 300 ,step=2)
    
    suffix = trial.suggest_categorical("normalization", ["min_max_init_grad", "mean_init_grad"])
    accuracy = trial.suggest_categorical("accuracy", [1e-5, 1e-6])
    
    # Other hyperparameters
    lr = trial.suggest_float("lr", 5e-6, 2e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 50, 600)

    activation_name = trial.suggest_categorical("activation", ["sigmoid", "relu", "leaky_relu"])
    if activation_name == "leaky_relu":
        activation_fn = nn.LeakyReLU()
    elif activation_name == "relu":
        activation_fn = nn.ReLU()
    elif activation_name == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation_name =="tanh":
        activation_fn = nn.Tanh()


    assert ROOT.exists(), f"Not found: {ROOT}"
    control_mesh_suffix =  "s100_100_100_b0_4000_0_5000_-4000_-0"
    basis_func_path = ROOT / "TrainingMapped" / control_mesh_suffix / "BasisFunctions" / f"basis_fts_matrix_{accuracy:.1e}{suffix}.npy"
    if 'init' in suffix.lower() and 'grad' in suffix.lower():
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
    
    # mask = ~(training_snapshots == 0).all(axis=(1, 2)) # omit indices that are not computed yet
    # training_snapshots       = training_snapshots[mask]
    # training_parameters      = training_parameters[mask, :]
    
    
    if PARAMETER_SPACE == "01":
        training_parameters[:, 0] = np.log10(training_parameters[:, 0])
        test_parameters[:, 0] = np.log10(test_parameters[:, 0])
    
    assert len(training_snapshots) == len(training_parameters)
    # Prepare data

    if 'init' in suffix.lower() and 'grad' in suffix.lower():
        training_snapshots  = training_snapshots[:, -1, :] # last time step
        test_snapshots      = test_snapshots[:, -1, :]
        logging.debug("Entered 'init' and 'grad' condition")
    elif 'init' in suffix.lower():
        training_snapshots  = training_snapshots[:, -1, :] -  training_snapshots[:, 0, :] 
        test_snapshots      = test_snapshots[:, -1, :] - test_snapshots[:, 0, :]
        logging.debug("Entered 'init' condition")
    else:
        training_snapshots  = training_snapshots[:, -1, :]
        test_snapshots      = test_snapshots[:, -1, :]
        logging.debug("Entered else statement condition")
    

    if "mean" in suffix.lower():
        scaling = Normalizations.Mean
    elif "min_max" in suffix.lower():
        scaling = Normalizations.MinMax
    else:
        raise ValueError("Invalid suffix.")
    logging.debug(f"Chose {scaling}")



    data_module = NirbDataModule(
        basis_func_mtrx=basis_functions,
        training_snaps=training_snapshots,
        training_param=training_parameters,
        val_param=test_parameters,
        val_snaps=test_snapshots,
        batch_size=batch_size,
        normalizer =scaling,
    )
    
    n_inputs = training_parameters.shape[1]
    n_outputs = basis_functions.shape[0]
    
    model = NirbModule(n_inputs, 
                       [hidden1] + hidden_layers_betw + [hidden6],
                       n_outputs,
                       activation=activation_fn,
                       learning_rate=lr)

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
            save_top_k=10,
            mode="min",
            filename = "{epoch:02d}-{Q2_val:.2e}",
            every_n_train_steps = 400
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
                        max_time={"minutes": 50},
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
    N_STEPS = 90_000 #100_000 #20_000
    # ACCURACY = 1e-5
    # SUFFIX = "min_max_init_grad"
    PARAMETER_SPACE = "03"
    ROOT = Path(__file__).parents[1] / "data" / PARAMETER_SPACE
    STUDY_NAME = "sweep"
    N_JOBS = 3
    N_TRIALS = 150


    assert ROOT.exists(), f"Not found: {ROOT}"
    db_path = ROOT / "db_total.sqlite3" #/f"db_{ACCURACY:.1e}{SUFFIX}.sqlite3"
    storage_param = {
        "storage": f"sqlite:///{db_path}",  # Specify the storage URL here.
        "study_name": STUDY_NAME,
        "load_if_exists": True
    }
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=30,
                                    n_warmup_steps=int(N_STEPS*0.33),
                                    n_min_trials=20
                                ),
                                **storage_param)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)