import logging
import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import optuna
from torch import nn
import numpy as np
import sys 
import logging
from datetime import datetime
sys.path.append(str(Path(__file__).parents[1]))
from src.pod.normalizer import match_scaler
from src.offline_stage import NirbModule, NirbDataModule, OptunaPruning
from src.utils import load_pint_data, find_snapshot_path, setup_logger, read_config

def objective(trial: optuna.Trial) -> float:
    # Architecture: variable number of layers and neurons per layer
    hidden1 = trial.suggest_int("hidden1", low = 2, high = 100)
    num_inbetw_layers = trial.suggest_int("num_inbetw_layers", 1, 4)
    hidden_layers_betw = [
        trial.suggest_int(f"hidden_layers_betw{i}", 50, 300, step=2)
        for i in range(num_inbetw_layers)
    ]
    hidden6 = trial.suggest_int("hidden6", low = 50,  high = 150 ,step=2)
    
    suffix = trial.suggest_categorical("scaler_output", ["none", "min_max"])
    scaler_features = trial.suggest_categorical("scaler_features", ["standardizer",  "min_max"])
    accuracy = 1e-5 #trial.suggest_categorical("accuracy", [1e-5, 1e-6])
    
    # Other hyperparameters
    lr = trial.suggest_float("lr", 5e-6, 2e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 15 , 100)

    activation_name = trial.suggest_categorical("activation", ["sigmoid", "leaky_relu", "tanh"])
    if activation_name == "leaky_relu":
        activation_fn = nn.LeakyReLU()
    elif activation_name == "relu":
        activation_fn = nn.ReLU()
    elif activation_name == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation_name =="tanh":
        activation_fn = nn.Tanh()

    if PROJECTION == "Mapped":
        basis_func_path = ROOT / "TrainingMapped" / control_mesh_suffix / f"BasisFunctions{FIELD_NAME}" / f"basis_fts_matrix_{accuracy:.1e}{suffix}.npy"
    else:
        basis_func_path = ROOT / f"BasisFunctions{FIELD_NAME}" / f"basis_fts_matrix_{accuracy:.1e}{suffix}.npy"


    assert basis_func_path.exists(), f"Basis function path {basis_func_path} does not exist."
    basis_functions         = np.load(basis_func_path)
    
    training_snapshots = np.load(find_snapshot_path(PROJECTION, suffix, FIELD_NAME, ROOT, control_mesh_suffix, "Training"))[:, -1, :]
    test_snapshots     = np.load(find_snapshot_path(PROJECTION, suffix, FIELD_NAME, ROOT, control_mesh_suffix, "Test"))[:, -1, :]
  
    train_param_path = ROOT / "training_samples.csv"
    test_param_path = ROOT / "test_samples.csv"
    training_parameters     = load_pint_data(train_param_path, is_numpy = True)
    test_parameters         = load_pint_data(test_param_path, is_numpy = True)
    
    if PARAMETER_SPACE == "09":
        zero_crossings = np.load(ROOT / "Exports/Training_zero_crossings.npy")
        mask = zero_crossings != 6
        training_snapshots = training_snapshots[mask, :] 
        training_parameters = training_parameters[mask, :]
    
    if PARAMETER_SPACE == "01":
        training_parameters[:, 0] = np.log10(training_parameters[:, 0])
        test_parameters[:, 0] = np.log10(test_parameters[:, 0])
    
    assert len(training_snapshots) == len(training_parameters)
    # Prepare data

    scaler_features = match_scaler(scaler_features)
    scaling_outputs = match_scaler(suffix)
    logging.info(f"{scaler_features=}")
    logging.info(f"{scaling_outputs=}")
    

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
                       scaler_features=str(scaler_features),
                       scaler_outputs=str(scaling_outputs))

    optuna_pruning = OptunaPruning(
        trial, 
        monitor="train_loss",        # or "val_loss"
        check_val_every_n_steps=20,
        mode="min",                  # we're minimizing loss
        )
    
    model_ckpt = ModelCheckpoint(
            monitor = "val_loss",
            save_last=True,
            save_top_k=2,
            mode="min",
            filename = "{epoch:02d}-{val_loss:.2e}",
            every_n_train_steps = 500
        )

    logger_dir_name = f"optuna_logs{FIELD_NAME}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(ROOT, name=logger_dir_name,
                               version=f"trial_{trial.number}_{timestamp}",
                               default_hp_metric=False)
    trainer = L.Trainer(max_epochs=N_EPOCHS,
                        enable_checkpointing=True,
                        logger=logger,
                        callbacks=[optuna_pruning, model_ckpt],
                        enable_progress_bar=False,
                        max_time={"minutes": 80},
                        )
    
    # model.basis_functions = data_module.basis_func_mtrx
    # model.val_snaps_scaled = data_module.val_snaps_scaled
    trainer.fit(model,
                train_dataloaders=data_module.train_dataloader(shuffle = True,
                                                                persistent_workers=True,
                                                                num_workers = N_JOBS + 1),
                val_dataloaders=data_module.validation_dataloader(num_workers = 1,
                                                                  persistent_workers=True)
    )
    
    # results = trainer.test(model, dataloaders=data_module.test_dataloader())
    # test_loss = results[0]['test_loss']
    train_loss = model.train_loss
    # trial.set_user_attr("test_loss", test_loss)
    val_loss = trainer.callback_metrics.get("val_loss")
    trial.set_user_attr("val_loss", float(val_loss))
    return float(train_loss)

if __name__ == "__main__":
    setup_logger(is_console=True, level=logging.INFO)
    N_EPOCHS = 90_000 #100_000 #20_000
    STUDY_NAME = "sweep"
    N_JOBS = 2
    N_TRIALS = 50
    
    config = read_config()
    ROOT = config['data_folder']
    assert ROOT.exists(), f"Not found: {ROOT}"
    FIELD_NAME = config['field_name']
    PROJECTION = config['projection']
    PARAMETER_SPACE = config['parameter_space']
    control_mesh_suffix =  config['control_mesh_suffix']
    



    assert ROOT.exists(), f"Not found: {ROOT}"
    # db_name = f"PS{PARAMETER_SPACE}_Optuna_{FIELD_NAME}" 
    # load_dotenv()
    # storage_param = {
    #     "storage": f"mysql+mysqlconnector://tsimader:{os.getenv("TSIMADER_SQL_PASSWORD")}@localhost/{db_name}",
    #     "study_name": STUDY_NAME,
    #     "load_if_exists": True
    # }
    db_path = ROOT / f"optuna_db{FIELD_NAME}.sqlite3" 
    storage_param = {
        "storage": f"sqlite:///{db_path}",  # Specify the storage URL here.
        "study_name": STUDY_NAME,
        "load_if_exists": True
    }
    
    
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=N_TRIALS * 0.33,
                                    n_warmup_steps=int(N_EPOCHS*0.33),
                                    n_min_trials=20
                                ),
                                **storage_param)
    
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)