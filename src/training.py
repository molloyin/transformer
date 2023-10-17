"""
Author:         Matthew Molloy
Date:           14/10/2023
Description:    Hyperparams, training loop, and evaluation logic for AI ./model.py
"""
import model  # Import the Transformer model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# Define model and data parameters
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# Initialize the Transformer model
transformer = model.Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data for training
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # Source data (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # Target data (batch_size, seq_length)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Set the model in training mode
transformer.train()

# Training loop for 100 epochs
for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])  # Forward pass
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))  # Calculate loss
    loss.backward()  # Backpropagate
    optimizer.step()  # Update model parameters
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")  # Print current epoch and loss

transformer.eval()
# Add token start-end vocabulary for sequence generation ("talk" to model)