import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import string

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


def plot_prediction(train_data, train_labels, test_data, test_labels, pred=None, fig_name=None):
    from datetime import datetime
    now = datetime.now()
    saving_fig_name = now.strftime("%y-%m-%d_%H_%M_%S")
    if fig_name is not None and isinstance(fig_name, str):
        saving_fig_name = fig_name
        
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="blue", s=4, label="training data")
    plt.scatter(test_data, test_labels, c="green", s=4, label='testing data')
    if pred is not None:
        plt.scatter(test_data, pred, c='red', s=4, label="Prediction")
        saving_fig_name = 'plot_pred_and_data' + saving_fig_name 
    elif pred is None:
        saving_fig_name = 'plot_data_' + saving_fig_name
    plt.legend(prop={"size":14})   
    plt.savefig(saving_fig_name)
    
plot_prediction(X_train, y_train, X_test, y_test)