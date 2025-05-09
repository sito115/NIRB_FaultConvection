import lightning as L
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
import optuna
from torch import nn
from torchinfo import summary
from optuna.trial import TrialState
import numpy as np
import sys 
sys.path.append(str(Path(__file__).parents[1]))
from scr.offline_stage import NirbModule, NirbDataModule, ComputeR2OnTrainEnd, MyEarlyStopping
from scr.utils import load_pint_data


def objective(trial: optuna.Trial) -> float:
    # Architecture: variable number of layers and neurons per layer
    hidden1 = trial.suggest_int("hiden1", low = 2, high = 50)
    num_inbetw_layers = trial.suggest_int("num_inbetw_layers", 1, 5)
    hidden_layers_betw = [
        trial.suggest_int(f"hidden_layers_betw{i}", 25, 300, step=2)
        for i in range(num_inbetw_layers)
    ]
    
    hidden6 = trial.suggest_int("hiden6", low = 10,  high = 100 ,step=2)
    
    N_EPOCHS = 25_000 #20_000
    ACCURACY = 1e-3
    
    # Other hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 10, 100)

    activation_name = "sigmoid" # trial.suggest_categorical("activation", ["relu", "leaky_relu", "sigmoid", "tanh"])
    if activation_name == "leaky_relu":
        activation_fn = nn.LeakyReLU()
    elif activation_name == "relu":
        activation_fn = nn.ReLU()
    elif activation_name == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation_name =="tanh":
        activation_fn = nn.Tanh()


    control_mesh_suffix =  "s100_100_100_b0_4000_0_5000_-4000_0"
    assert ROOT.exists(), f"Not found: {ROOT}"
    basis_func_path = ROOT / "TrainingMapped" / control_mesh_suffix / "BasisFunctions" / f"basis_fts_matrix_{ACCURACY:.1e}.npy"
    train_snapshots_path = ROOT / "TrainingMapped" / control_mesh_suffix / "Exports" / "Training_temperatures.npy"
    test_snapshots_path = ROOT / "TestMapped" / control_mesh_suffix / "Exports" / "Test_temperatures.npy"
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
    
    # summary(model.model, 
    #         input_size=(1, n_inputs),
    #         col_names=["input_size",
    #                    "output_size",
    #                    "num_params"],)
            
    optuna_pruning = MyEarlyStopping(
        trial, 
        min_epochs=15_000,
        monitor="train_loss",        # or "val_loss"
        mode="min",                  # we're minimizing loss
        )
    
    r2_callback = ComputeR2OnTrainEnd(data_module.training_param_scaled,
                                      data_module.training_snaps_scaled,
                                      data_module.basis_func_mtrx)


    # logger = TensorBoardLogger(ROOT / "Optuna_Logs_2", name=f"trial_{trial.number}")
    trainer = L.Trainer(max_epochs=N_EPOCHS,
                        logger=False,
                        enable_checkpointing=False,
                        callbacks=[optuna_pruning, r2_callback],
                        enable_progress_bar=False,
                        max_time={"minutes": 50},
                        )

    trainer.fit(model, train_dataloaders=data_module.train_dataloader(shuffle=True))
    
    # results = trainer.test(model, dataloaders=data_module.train_dataloader())
    # test_loss = results[0]['test_loss']
    # train_loss = trainer.callback_metrics["train_loss"].item()
    train_loss = model.train_loss
    return train_loss

if __name__ == "__main__":
    ROOT = Path(__file__).parents[1] / "data" / "03"
    db_path = ROOT / "db.sqlite3"
    storage_param = {
        "storage": f"sqlite:///{db_path}",  # Specify the storage URL here.
        "study_name": "optuna_sweep_1",
        "load_if_exists": True
    }
    study = optuna.create_study(direction="minimize", **storage_param)
    study.optimize(objective, n_trials=400, n_jobs=5)

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