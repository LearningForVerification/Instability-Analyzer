import torch
import torchvision
import numpy as np
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torchvision.transforms as tr
import torch.optim as optim
import onnx
import json

from generate_nn_dims import get_dims

# Constants
NETWORK_FOLDER = "networks"
DATASET_DIR = 'files'

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Neural Network definition
class NN(nn.Module):
    def __init__(self, hidden_size):
        super(NN, self).__init__()
        self.l1 = nn.Linear(784, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input if necessary
        x = self.relu(self.l1(x))
        x = self.l2(x)
        return x


# Function to check accuracy
def check_accuracy(loader, model):
    if loader.dataset.dataset.train:
        print("Accuracy on training data")
    else:
        print("Accuracy on testing data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.view(x.size(0), -1)  # Flatten the input if necessary

            scores = model(x)
            _, predictions = scores.max(1)

            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

        accuracy = num_correct / num_samples * 100
        print(f'Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}')

    model.train()
    return accuracy


# Hyperparameters
hidden_layer_dims, threshold = get_dims()
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.95
num_epochs = 1
log_interval = 10
input_size = 784

# Setting random seed
torch.manual_seed(1)

# Data loading and transformations
transform = tr.Compose([
    tr.ToTensor(),
    tr.Normalize(mean=0.5, std=0.5),
    tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image
])

train_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=False, download=True, transform=transform)

# Creating subsets and data loaders
train_subset = Subset(train_dataset, list(range(4000)))
train_loader = DataLoader(train_subset, batch_size=batch_size_train, shuffle=True)

test_subset = Subset(test_dataset, list(range(5000)))
test_loader = DataLoader(test_subset, batch_size=batch_size_test, shuffle=False)

# Loss and Accuracy tracking
loss_acc = {}

# Criterion and optimizer setup
criterion = nn.CrossEntropyLoss()

# Training loop
for h_dim in hidden_layer_dims:
    model = NN(h_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Determine if scheduler should be used based on param_number
    param_number = ((input_size + 1) * h_dim + (h_dim + 1) * 10)
    if param_number <= threshold:
        decrease_epoch = True
        scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
    else:
        decrease_epoch = False

    # Epoch training
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Adjust learning rate if using scheduler
            if decrease_epoch:
                scheduler.step()

            # Print training progress
            if batch_idx % log_interval == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Early stopping condition if loss is very low
        if decrease_epoch and torch.allclose(loss, torch.tensor(0.0)):
            break

    # Calculate accuracy on test and train sets
    test_acc = check_accuracy(test_loader, model)
    train_acc = check_accuracy(train_loader, model)
    loss_acc[h_dim] = [test_acc, train_acc, loss.item()]

    # Save model in PyTorch format
    torch.save(model.state_dict(), f"{NETWORK_FOLDER}/torch_format/model_{h_dim}.pth")

    # Export model to ONNX format
    onnx_path = f"{NETWORK_FOLDER}/onnx_format/model_{h_dim}.onnx"
    dummy_input = torch.randn(1, 784, device=device)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=False)

    # Check the exported ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Save loss_acc dictionary to a file
    with open(f"{NETWORK_FOLDER}/loss_acc.json", "w") as json_file:
        json.dump(loss_acc, json_file)

