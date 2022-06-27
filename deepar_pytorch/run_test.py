from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from deepar.dataset.time_series import MockTs
from deepar.model.lstm import DeepAR
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from deepar_pytorch import loss
import pdb
import torch
import torch.functional as f
from torch import nn
import pytorch_lightning as pl
import numpy as np
from deepar_pytorch.TS_Dataset import *
# Get cpu or gpu device for training.
from deepar_pytorch.model import NeuralNetwork_pl
{'n_layers': 5, 'dropout': 0.3299530064538784, 'seq_len': 59, 'output_features': 1, 'batch_size': 819}

ts = MockTs(dimensions=1, resolution=1.0, t_min=0, t_max=10000, n_steps=100, divisor=10.0)  # you can change this for multivariate time-series!
df = pd.DataFrame({"t":np.arange(ts.t_min, ts.t_max), "value":ts.timepoint(np.arange(ts.t_min, ts.t_max))})
r1 = TS_Dataset(df, x_cols=["t","value"], y_col="value", sequence_length=59)
# logger = TensorBoardLogger(save_dir="logs")
hparams = dict(train_dataset=r1, input_features=len(r1.x_cols),output_features=1,n_hidden_layers=5,dropout=0.3, lr=0.001, batch_size=819)
model2 = NeuralNetwork_pl(**hparams)
checkpoint_callback = ModelCheckpoint(dirpath="logs/", save_top_k=2, monitor="val_loss")
trainer = pl.Trainer(gradient_clip_val = 0.0,
                     log_every_n_steps=1,
                     min_epochs=10, max_epochs=1000,
                     auto_lr_find=True,
                     auto_scale_batch_size=False,
                     track_grad_norm=2,
                     callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30, min_delta=0.01), checkpoint_callback],
                     terminate_on_nan=False
                     )


trainer.tune(model2)
print(model2.lr)
trainer.fit(model2, val_dataloaders=model2.train_dataloader())

model2 = model2.load_from_checkpoint(checkpoint_path= checkpoint_callback.best_model_path, **hparams)
trainer.validate(model2, model2.train_dataloader())
device = "cpu"
X,y = ts.next_batch(1,400)
X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
X, y = X.to(device), y.to(device)
y_pred =model2.forward(X)[0]
plt.plot(y_pred.reshape(-1).detach().numpy(), label = "pred")
plt.plot(y.reshape(-1), label = "real")
plt.plot(X[:,:,0].reshape(-1), label = "x")
plt.legend()

model2.debug =True
# trainer.validate(model2, model2.train_dataloader(), verbose=True)
