import torch
import torch.nn as nn
from n_solush_dataset import RubarrelDataset
import torch.optim as optim
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())  # You can use other activation functions as well
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())  # You can use other activation functions as well
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Example usage:
input_size = 24
output_size = 10
hidden_size = 64
num_layers = 5

# Initialize the MLP
mlp = MLP(input_size, output_size, hidden_size, num_layers)

# Instantiate the Rubarreldataset class
n_sample = 100000  # Example number of samples
n_target = 10    # Example target size
dataset = RubarrelDataset(n_sample, n_target)

# Create DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the MLP
input_size = 24
output_size = n_target
mlp = MLP(input_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = mlp(inputs.float())  # Convert inputs to float since the MLP expects float inputs
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")# Should be torch.Size([32, 10])
