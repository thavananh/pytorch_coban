from ast import List
from turtle import forward
from sympy import true
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
from torchinfo import summary
from datetime import datetime
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.weights * x + self.bias
    
class LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)    
        

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

def saving_model(model, sample_input=None, output_model_format_type="torch_default", saving_model_folder="models", output_model_name="default_model_name"):
    model_map = {
        'torch_default': '.pth',
        'torch_script': '.pth',
        'safetensors': '.safetensors',
        'onnx': '.onnx'
    }
    now = datetime.now().strftime("%y-%m-%d_%H_%M_%S")
    if saving_model_folder == "datetime":
        print("folder name now set to datetime this script run")
        saving_model_folder = now
    model_path = Path(saving_model_folder)
    model_path.mkdir(parents=True, exist_ok=True)

    model_save_path = None
    if not isinstance(output_model_name, str):
        print('your output model is not valid, fall back to default output model name')
        output_model_name = 'default_model_name'
    if output_model_name == "default_model_name":
        output_model_name = "model_" + now
    if not isinstance(output_model_format_type, str) or output_model_format_type not in model_map:
        print('Your output model format type not valid, fall back to default torch native saving format')
        output_model_format_type = 'torch_default'
    ext = model_map.get(output_model_format_type)
    if ext is not None:
        output_model_name = output_model_name + ext
    else:
        print('Your format not valid, fall back to default torch native saving format')
        output_model_name = output_model_name + '.pth'
        output_model_format_type = 'torch_default'
    model_save_path = model_path / output_model_name
    if output_model_format_type == 'torch_default':
        print(f'Your model is saving to {model_save_path}')
        torch.save(model.state_dict(), model_save_path)
    elif output_model_format_type == 'torch_script':
        scripted_model = torch.jit.script(model)
        print(f'Your model is saving to {model_save_path} as a TorchScript model')
        torch.jit.save(scripted_model, model_save_path)
    elif output_model_format_type == 'onnx':
        print(f'Your model is saving to {model_save_path} as a ONNX model')

        if sample_input is None:
            exported_args = (torch.rand(1, 1),)
        elif sample_input is isinstance(sample_input, torch.Tensor):
            exported_args = (sample_input,)
        elif sample_input is isinstance(sample_input, (tuple, list)):
            exported_args = tuple(sample_input,)
        else:
            exported_args = (torch.rand(1,1),)
            
        # This export is legacy and it will be deprecated in torch 2.9. Right now, pytorch is only 2.8 but i will adapt it to the newer method
        # torch.onnx.export(
        #     model,
        #     sample_input,
        #     model_save_path,
        #     export_params=true,
        #     opset_version=11,
        #     do_constant_folding=True,
        #     input_names=['input'],
        #     output_names=['output'],
        #     dynamic_shapes={"input": {0: "batch"}, "output": {0: "batch"}},
        # )
            
        torch.onnx.export(
            model,
            exported_args,
            model_save_path,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"x": {0: "batch"}},
            dynamo=True
        )
    elif output_model_format_type == 'safetensors':
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict_cpu, model_save_path)
        

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
    # model_0 = LinearRegressionModel()
    model_0 = LinearRegressionV2()
    print("Model summary")
    summary(model_0, X_train.shape)
    train_model(model_0, X_train, y_train, X_test, y_test, 1000)
    saving_model(model=model_0, output_model_format_type="safetensors", sample_input=X_test, saving_model_folder="datetime")

if __name__ == "__main__":
    main()