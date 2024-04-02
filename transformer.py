import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from n_solush_dataset import RubarrelDataset


# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, target_size, hidden_size=128, num_layers=2, num_heads=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding_src = nn.Embedding(input_size, hidden_size)
        self.embedding_tgt = nn.Embedding(target_size, hidden_size)  # Different embedding for target
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, target_size)

    def forward(self, src, tgt):
        print("s", src.size(), tgt.size())
        src = self.embedding_src(src)
        tgt = self.embedding_tgt(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output


# Generate some sample data
# Replace this with your actual dataset loading
def generate_sample_data(num_samples, input_size, target_size):
    data = []
    for _ in range(num_samples):
        input_vector = np.random.randint(0, input_size, size=24)
        target_vector = np.random.randint(0, target_size, size=10)
        data.append((input_vector, target_vector))
    return data


# Data parameters
input_size = 24  # example size
target_size = 10  # example size
batch_size = 32
num_samples = 10000


# Create dataset and dataloader
dataset = RubarrelDataset(n_samples=num_samples, n_actions=target_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = TransformerModel(input_size, target_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs, targets)  # Ignore last target for input
        loss = criterion(outputs, targets)  # Target shifted by 1
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Evaluation
model.eval()
with torch.no_grad():
    # Sample inference
    sample_input = torch.randint(0, input_size, (1, 24))
    sample_target = torch.randint(0, target_size, (1, 10))
    output = model(sample_input, sample_target)
    print("Sample output:", output)