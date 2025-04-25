from E_coefficients_model import NirbModule, NirbDataModule
from helpers import load_pint_data
import lightning as L
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
import optuna
from torch import nn
from torchinfo import summary
from optuna.trial import TrialState
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping

def objective(trial):
    # Architecture: variable number of layers and neurons per layer
    # num_layers = trial.suggest_int("num_layers", 2, 5)
    # hidden_layers = [
    #     trial.suggest_categorical(f"n_units_l{i}", [16, 32, 64, 128])
    #     for i in range(num_layers)
    # ]
    
    # hidden3 = trial.suggest_categorical("hiden3", [32, 64, 128])
    hidden4 = trial.suggest_categorical("hiden4", [64, 128])
    # hidden5 = trial.suggest_categorical("hiden5", [32, 64, 128])
    # hidden6 = trial.suggest_categorical("hiden6", [16, 32, 64])
    
    # hidden_layers = [8, 16] + [hidden3, hidden4, hidden5, hidden6]

    ROOT = Path("/Users/thomassimader/Documents/NIRB/Snapshots/01")
    N_EPOCHS = 10_000
    
    # Other hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 5e-2)
    
    param_path = ROOT / "training_samples.csv"
    basis_func_path = ROOT / "BasisFunctions" / "basis_fts_matrix.npy"
    snapshots_path = ROOT / "Exports" / "temperatures.npy"
    assert param_path.exists(), f"Does not exit: {param_path}"
    assert basis_func_path.exists(), f"Does not exit: {basis_func_path}"
    assert snapshots_path.exists(),f"Does not exit: {snapshots_path}"
    
    batch_size = 20 # trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    activation_name = trial.suggest_categorical("activation", ["relu", "sigmoid"])
    if activation_name == "relu":
        activation_fn = nn.ReLU()
    elif activation_name == "sigmoid":
        activation_fn = nn.Sigmoid()


    ROOT = Path(__file__).parent.parent / "Snapshots" / "01"
    assert ROOT.exists(), f"Not found: {ROOT}"
    

    basis_func_path = ROOT / "BasisFunctions" / "basis_fts_matrix.npy"
    train_snapshots_path = ROOT / "Exports" / "Training_temperatures.npy"
    test_snapshots_path = ROOT / "Exports" / "Test_temperatures.npy"
    train_param_path = ROOT / "training_samples.csv"
    test_param_path = ROOT / "test_samples.csv"
    
    
    basis_functions         = np.load(basis_func_path)
    training_snapshots      = np.load(train_snapshots_path)
    training_parameters     = load_pint_data(train_param_path, is_numpy = True)
    # test_snapshots          = np.load(test_snapshots_path)
    # test_parameters         = load_pint_data(test_param_path, is_numpy = True)
    
    # Prepare data
    training_snapshots = training_snapshots[:, -1, :] # last time step
    training_parameters = training_parameters[:, 2:] 
    # test_snapshots = test_snapshots[:, -1, :] # last time step
    # test_parameters = test_parameters[:, 2:] 
     
     
    dm = NirbDataModule(
        basis_func_mtrx=basis_functions,
        training_snaps=training_snapshots,
        training_param=training_parameters,
        batch_size=batch_size,
    )
    
    n_inputs = training_parameters.shape[1]
    n_outputs = basis_functions.shape[0]
    
    
    model = NirbModule(n_inputs, [8, 16, 32, hidden4, 32],
                       n_outputs,
                       activation=activation_fn,
                       learning_rate=lr)
    
    summary(model.model, 
            input_size=(1, n_inputs),
            col_names=["input_size",
                       "output_size",
                       "num_params"],)
            
    early_stop = EarlyStopping(
        monitor="train_loss",        # or "val_loss"
        stopping_threshold=1e-8,      # 💥 stop when loss drops below this
        divergence_threshold=10_000,
        mode="min",                  # we're minimizing loss
        verbose=True,
        patience=10_000,
        check_on_train_epoch_end=True
        )
    
    logger = TensorBoardLogger(ROOT, name="nn_logs_opti_new")
    trainer = L.Trainer(max_epochs=N_EPOCHS,
                        logger=logger,
                        log_every_n_steps=100,  # Reduce logging frequency
                        # enable_checkpointing=True,
                        callbacks=[early_stop], #, RichProgressBar(refresh_rate=BATCH_SIZE, leave=False)],
                        # precision=16,
                        # max_time={"days": 0, "hours": 0, "minutes": 25},
                        # strategy='ddp',
                        enable_progress_bar=False,
                        profiler="simple",
                        # devices=3,
                        # accelerator= "cpu" #'mps',
                        )

    trainer.fit(model, train_dataloaders=dm.train_dataloader())
    results = trainer.test(model, dataloaders=dm.train_dataloader())
    test_loss = results[0]['test_loss']
    # return trainer.callback_metrics["train_loss"].item()
    return test_loss

if __name__ == "__main__":
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

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