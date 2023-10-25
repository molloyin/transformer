"""
Author:         Matthew Molloy
Date:           14/10/2023
Description:    Hyperparams, training loop, and evaluation logic for AI ./model.py
"""
import scratchmodel as model  # Import the Transformer model
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as plt

# define model and data parameters
tgt_vocab_size = 2000
d_model = 1024
num_heads = 8
num_layers = 5
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = model.DecoderTransformer(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# generate random sample data for training
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # Target data (batch_size, seq_length)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# set the model in training mode
transformer.train()
losses = []

# training loop for 100 epochs
for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(tgt_data[:, :-1])  # forward pass
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))  # Calculate loss
    loss.backward()  # backpropagate
    optimizer.step()  # update model parameters
    
    losses.append(loss.item())
    print(f"Training Epoch: {epoch+1}, Loss: {loss.item()}")  # print current epoch and loss

transformer.eval()
# add token start-end vocabulary for sequence generation ("talk" to model)