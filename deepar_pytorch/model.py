import pdb

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from deepar_pytorch import loss


class NeuralNetwork_pl(pl.LightningModule):
    def __init__(self, input_features, output_features, n_hidden_layers, dropout,train_dataset, loss = loss.gausian_prob, lr = 0.001, batch_size = 50):
        self.debug = False
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.lr = lr
        print("applied init")
        self.output_features = output_features
        super(NeuralNetwork_pl, self).__init__()
        self.rnn = nn.LSTM(input_size=input_features, hidden_size=output_features, num_layers=n_hidden_layers,
                           dropout=dropout, batch_first=True)
        self.W_mu = nn.Linear(output_features, 1)
        self.W_sig = nn.Linear(output_features, 1)
        self.loss = loss

    def forward(self, x):
        # pdb.set_trace()
        batch_size = x.shape[0]
        x = self.rnn(x)[0].reshape(-1, self.output_features)
        mu = self.W_mu(x)
        sigma = nn.functional.softplus(self.W_sig(x))
        return (mu.reshape(batch_size, -1, 1), sigma.reshape(batch_size, -1, 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss_value = self.loss(y_hat, y)
        self.log
        return {"loss": loss_value}




    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if self.debug:
            pdb.set_trace()
        self.log("val_loss1", loss, on_step=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs):

        self.log("val_loss",  np.array(validation_step_outputs).mean(), on_epoch=True)
        return {"val_loss": np.array(validation_step_outputs).mean()}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
