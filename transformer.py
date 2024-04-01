import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from n_solush_dataset import RubarrelDataset


def collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)
    return padded_inputs, padded_targets

dataset = RubarrelDataset(32000, 5)
class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, num_layers, hidden_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder = nn.TransformerEncoderLayer(embedding_dim, dim_feedforward=hidden_size, nhead=8)
        self.decoder = nn.Linear(embedding_dim, output_vocab_size)

    def forward(self, input_tensor):
        embedded_input = self.embedding(input_tensor)
        encoded_input = self.encoder(embedded_input)
        output_tensor = self.decoder(encoded_input)
        return output_tensor

# Step 3: Define your model, criterion, and optimizer
input_vocab_size = 5  # Adjust based on your input data
output_vocab_size = 4  # Adjust based on your output data
embedding_dim = 64
num_layers = 4
hidden_size = 128

model = TransformerModel(input_vocab_size, output_vocab_size, embedding_dim, num_layers, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Create DataLoader for batching and shuffling data
batch_size = 32  # Adjust based on your memory constraints and dataset size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Step 5: Training loop
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_data, target_data = batch
        output = model(input_data)
        loss = criterion(output.view(-1, output_vocab_size), target_data.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(input_data)

    # Print average loss per epoch
    average_loss = total_loss / len(dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}')

# Step 6: Evaluation
# Evaluate the trained model on a separate test set if available
