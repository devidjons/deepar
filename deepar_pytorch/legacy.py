import matplotlib.pyplot as plt
import torch
from torch import nn

# Get cpu or gpu device for training.
from deepar.dataset.time_series import MockTs
from deepar_pytorch import loss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_features, output_features, n_hidden_layers, dropout):
        self.output_features = output_features
        super(NeuralNetwork, self).__init__()
        self.rnn= nn.LSTM(input_size=input_features, hidden_size=output_features, num_layers=n_hidden_layers, dropout=dropout, batch_first=True)
        self.l1 = nn.Linear(output_features, output_features)
        self.act = nn.ReLU()
        self.W_mu = nn.Linear(output_features, 1)
        self.W_sig = nn.Linear(output_features, 1)

    def forward(self, x):
        # pdb.set_trace()
        batch_size = x.shape[0]
        x = self.rnn(x)[0].reshape(-1, self.output_features)
        x = self.act(self.l1(x))
        mu = self.W_mu(x)
        sigma = torch.exp(1+self.W_sig(x))+1e-6
        return (mu.reshape(batch_size,-1,1), sigma.reshape(batch_size,-1,1))


    def predict(self, x):
        return self.forward(torch.from_numpy(x).float())









def train(dataloader, model, loss_fn, optimizer, epochs, batch_size = 100, backward_size = 100):
    for epoch in range(epochs):
        for batch in range((dataloader.t_max - dataloader.t_min)//batch_size):
            X,y = dataloader.next_batch(batch_size, n_steps=backward_size)
            X,y =torch.from_numpy(X).float(), torch.from_numpy(y).float()
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            # pdb.set_trace()
            loss = loss_fn(pred, y)
            # pdb.set_trace()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X,y = dataloader.next_batch(1,1000)
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        loss = loss.item()
        print(f"loss: {loss:>7f}  [{epoch:>5d}]")





ts = MockTs(dimensions=1, resolution=1.0, t_min=0, t_max=10000, n_steps=100, divisor=10.0)  # you can change this for multivariate time-series!

model = NeuralNetwork(input_features=2,output_features=3,n_hidden_layers=2,dropout=0.1).to(device)
loss_fn = loss.gausian_prob
optimizer = torch.optim.Adam(model.parameters())
train(ts, model, loss_fn, optimizer, 150, backward_size=100)
X,y = ts.next_batch(1,400)
X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
X, y = X.to(device), y.to(device)
y_pred =model.forward(X)[0]
plt.plot(y_pred.reshape(-1).detach().numpy(), label = "pred")
plt.plot(y.reshape(-1), label = "real")
plt.plot(X[:,:,0].reshape(-1), label = "x")
plt.legend()