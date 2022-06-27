import argparse
import pandas as pd
import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from deepar.dataset.time_series import MockTs
from deepar_pytorch.TS_Dataset import TS_Dataset
from deepar_pytorch.model import NeuralNetwork_pl
import numpy as np

def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_hiden_layers = trial.suggest_int("n_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.05, 0.5, log = True)
    seq_len= trial.suggest_int("seq_len", 10, 100, log=True)
    output_features =trial.suggest_int("output_features", 1, 100, log= True)
    batch_size = trial.suggest_int("batch_size", 100, 5000, log= True)
    hyperparameters = dict(n_hiden_layers=n_hiden_layers, dropout=dropout, seq_len=seq_len,output_features = output_features,batch_size = batch_size )
    print(hyperparameters)

    ts = MockTs(dimensions=1, resolution=1.0, t_min=0, t_max=10000, n_steps=100,
                divisor=10.0)  # you can change this for multivariate time-series!
    df = pd.DataFrame({"t": np.arange(ts.t_min, ts.t_max), "value": ts.timepoint(np.arange(ts.t_min, ts.t_max))})
    r1 = TS_Dataset(df, x_cols=["value"], y_col="value", sequence_length=seq_len)
    # logger = TensorBoardLogger(save_dir="logs")
    checkpoint_callback = ModelCheckpoint(dirpath="logs/", save_top_k=2, monitor="val_loss")
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=30, min_delta=0.01)
    model = NeuralNetwork_pl(train_dataset=r1,
                              input_features=len(r1.x_cols),
                              output_features=output_features,
                              n_hidden_layers=n_hiden_layers,
                              dropout=dropout, lr=0.001, batch_size=batch_size)
    trainer = pl.Trainer(gradient_clip_val=0.0,
                         min_epochs=10, max_time={"minutes":5},
                         auto_lr_find=True,
                         auto_scale_batch_size=False,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         enable_checkpointing=True,
                         enable_progress_bar=False
                         )

    trainer.tune(model)
    if model.lr is None:
        model.lr = 0.001

    hyperparameters = dict(n_hiden_layers=n_hiden_layers, dropout=dropout, seq_len=seq_len,output_features = output_features,batch_size = batch_size )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, val_dataloaders = model.train_dataloader())
    model = model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path, train_dataset=r1,
                              input_features=len(r1.x_cols),
                              output_features=output_features,
                              n_hidden_layers=n_hiden_layers,
                              dropout=dropout, lr=0.001, batch_size=batch_size)
    trainer.validate(model, model.train_dataloader())

    return trainer.callback_metrics["val_loss"].item()


pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=80, timeout=6*3600, n_jobs=4, show_progress_bar=True)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))