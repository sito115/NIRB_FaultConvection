from E_coefficients_model import NirbModule, NirbDataModule, ComputeR2OnTrainEnd
from helpers import load_pint_data
import lightning as L
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import optuna
from torch import nn
from torchinfo import summary
from optuna.trial import TrialState
import numpy as np
from helpers import MyEarlyStopping

def objective(trial: optuna.Trial) -> float:
    # Architecture: variable number of layers and neurons per layer
    num_inbetw_layers = trial.suggest_int("num_inbetw_layers", 1, 5)
    hidden_layers_betw = [
        trial.suggest_int(f"hidden_layers_betw{i}", 10, 300, step=5)
        for i in range(num_inbetw_layers)
    ]
    
    hidden1 = trial.suggest_int("hiden1", low = 5, high = 20)
    # hidden2 = trial.suggest_int("hiden2", low = 16,  high = 64 ,step=4)
    # hidden3 = trial.suggest_int("hiden3", low = 32,  high = 128 ,step=8)
    # hidden4 = trial.suggest_int("hiden4", low = 32,  high = 128 ,step=8)
    
    # is_hidden5 = trial.suggest_categorical("is_hidden5", [True, False])
    # if is_hidden5:
    #     hidden5 = trial.suggest_int("hiden5", low = 32,  high = 128 ,step=8)
    hidden6 = trial.suggest_int("hiden6", low = 10,  high = 64 ,step=2)
    
    # hidden_layers = [8, 16] + [hidden3, hidden4, hidden5, hidden6]

    ROOT = Path("/Users/thomassimader/Documents/NIRB/Snapshots/01")
    N_EPOCHS = 25_000
    ACCURACY = 1e-3
    
    # Other hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 2.5e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 10, 25)

    activation_name =  "sigmoid" #  trial.suggest_categorical("activation", ["relu", "sigmoid"])
    if activation_name == "relu":
        activation_fn = nn.ReLU()
    elif activation_name == "sigmoid":
        activation_fn = nn.Sigmoid()


    ROOT = Path(__file__).parent.parent / "Snapshots" / "01"
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
        test_snaps=test_snapshots,
        test_param=test_parameters,
        batch_size=batch_size,
    )
    
    n_inputs = training_parameters.shape[1]
    n_outputs = basis_functions.shape[0]
    
    
    model = NirbModule(n_inputs, 
                       [hidden1] + hidden_layers_betw + [hidden6],
                       n_outputs,
                       activation=activation_fn,
                       learning_rate=lr)
    
    summary(model.model, 
            input_size=(1, n_inputs),
            col_names=["input_size",
                       "output_size",
                       "num_params"],)
            
    early_stop = MyEarlyStopping(
        trial, 
        monitor="train_loss",        # or "val_loss"
        mode="min",                  # we're minimizing loss
        )
    
    r2_callback = ComputeR2OnTrainEnd(data_module.training_param_scaled,
                                      data_module.training_snaps_scaled,
                                      data_module.basis_func_mtrx)


    logger = TensorBoardLogger(ROOT / "Optuna_Logs", name=f"trial_{trial.number}")
    trainer = L.Trainer(max_epochs=N_EPOCHS,
                        min_epochs=5000,
                        logger=logger, #logger,
                        # enable_checkpointing=True,
                        callbacks=[early_stop, r2_callback], #, RichProgressBar(refresh_rate=BATCH_SIZE, leave=False)],
                        # strategy='ddp',
                        enable_checkpointing = True,
                        enable_progress_bar=False,
                        profiler="simple",
                        # devices=3,
                        # accelerator= "cpu" #'mps',
                        )

    trainer.fit(model, train_dataloaders=data_module.train_dataloader())
    
    results = trainer.test(model, dataloaders=data_module.train_dataloader())
    test_loss = results[0]['test_loss']
    # return trainer.callback_metrics["train_loss"].item()
    return test_loss

if __name__ == "__main__":
    
    # storage = optuna.storages.InMemoryStorage()
    storage_param = {
        "storage":"sqlite:///db.sqlite3",  # Specify the storage URL here.
        "study_name": "optuna_sweep",
        "load_if_exists": True
    }
    study = optuna.create_study(direction="minimize", **storage_param)
    study.optimize(objective, n_trials=400, n_jobs=3)

    print("Best hyperparameters:")
    print(study.best_params)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))