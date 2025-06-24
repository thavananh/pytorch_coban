from ast import List
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
from torchinfo import summary
from datetime import datetime

def plot_prediction(train_data, train_labels, test_data, test_labels, pred=None, fig_name=None):
        
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="blue", s=4, label="training data")
    plt.scatter(test_data, test_labels, c="green", s=4, label='testing data')
    now = datetime.now()
    saving_fig_name = now.strftime("%y-%m-%d_%H_%M_%S")
    if fig_name is not None and isinstance(fig_name, str):
        saving_fig_name = fig_name
    if pred is not None:
        plt.scatter(test_data, pred, c='red', s=4, label="Prediction")
        saving_fig_name = 'plot_pred_and_data' + saving_fig_name 
    elif pred is None:
        saving_fig_name = 'plot_data_' + saving_fig_name
    plt.legend(prop={"size":14})   
    plt.savefig(saving_fig_name)
    # plt.show()
    # plt.close()

def plot_train_test_loss(epochs_counts, train_loss_values, test_loss_values, fig_name):
    plt.figure(figsize=(10,7))
    plt.scatter(epochs_counts, train_loss_values, label="Train loss")
    plt.scatter(epochs_counts, test_loss_values, label="Test loss")
    now = datetime.now()
    saving_fig_name = now.strftime("%y-%m-%d_%H_%M_%S")
    if fig_name is not None and isinstance(fig_name, str):
        saving_fig_name = fig_name
    plt.legend(prop={"size":10})
    plt.savefig(saving_fig_name)
    # plt.show()
    # plt.close()
 
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.weights * x + self.bias

def train_model(model, train_data, train_labels, test_data, test_labels, num_epochs):
    train_loss_values = []
    test_loss_values = []
    epoch_counts = []
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    print("Model weights before train")
    print(model.state_dict())
    for epoch in tqdm(range(num_epochs)):
        model.train()
        y_pred = model(train_data)
        train_loss = loss_fn(y_pred, train_labels)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_pred = model(test_data)
            test_loss = loss_fn(test_pred, test_labels)
            epoch_counts.append(epoch)
            train_loss_values.append(train_loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())                
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | MAE Train Loss: {train_loss} | MAE Test Loss: {test_loss}")
    
    print("Model weights after train")
    print(model.state_dict())
    with torch.inference_mode():
        plot_prediction(train_data, train_labels, test_data, test_labels, pred=model(test_data))
        
    plot_train_test_loss(epoch_counts, train_loss_values, test_loss_values, "train_test_loss_chart")


        

def main():
    print("PyTorch version:", torch.__version__)
    weight = 0.7
    bias = 0.3
    start = 0
    end = 1
    step= 0.02
    X = torch.arange(start, end,step).unsqueeze(dim=1)
    y = weight * X + bias
    print("Data shape:", X.shape)
    print("Data:", X)
    print("Labels shape:", y.shape)
    print("Labels:", y)

    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    print(len(X_train), len(y_train), len(X_test), len(y_test))
    plot_prediction(X_train, y_train, X_test, y_test)
    
    seed = 42
    print(f"starting seed model with seed {seed}")
    print(X_train)
    torch.manual_seed(seed)
    model_0 = LinearRegressionModel()
    print("Model summary")
    summary(model_0, X_train.shape)
    train_model(model_0, X_train, y_train, X_test, y_test, 1000)

    
    

if __name__ == "__main__":
    main()