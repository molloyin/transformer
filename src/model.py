"""
Author:         Matthew Molloy
Date:           14/10/2023
Description:    Define and combine modules for decoder only transformer
"""

import math
import torch
import torch.nn as nn

# defining masked multi head attn as a neural network module
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, tensor_dimensions):
        super(MaskedMultiHeadAttention, self).__init__()    # inheret behaviour of nn.Module
        assert tensor_dimensions % num_heads == 0           # having remainder heads is a mess
        self.n_h = num_heads
        self.t_d = tensor_dimensions
        self.d_k = tensor_dimensions // num_heads           # dimensions of query/key/values

        # linear transformation modules for attn score distribution calc (how well does query match key)
        self.w_query = nn.Linear(self.t_d, self.t_d)
        self.w_key = nn.Linear(self.t_d, self.t_d)       
        self.w_val = nn.Linear(self.t_d, self.t_d)
        self.w_out = nn.Linear(self.t_d, self.t_d)

    def forward(self, Q, K, V, mask):
        batch_size = Q.size()

        # split heads
        query = Q.view(batch_size, batch_size, self.n_h, self.t_d).transpose(1, 2)
        key = K.view(batch_size, batch_size, self.n_h, self.t_d).transpose(1, 2)
        value = V.view(batch_size, batch_size, self.n_h, self.t_d).transpose(1, 2)

        # compute attn scores, apply mask and softmax
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # set masked values to very low score
        attn_prob_distribution = torch.softmax(attn_scores, dim=1)
        attn_output = torch.matmul(attn_prob_distribution, value)
        # combine heads, reshape, and return concatonated + linearly transformed attention
        return self.w_out(attn_output.transpose(1, 2).contiguous().view(batch_size, batch_size, self.t_d))


# introduce non-linearity and capture position-specific patterns
class PositionWiseFF(nn.Module):
    def __init__(self, tensor_dimensions, ff_dimensions):
        super(PositionWiseFF, self).__init__()
        self.lt1 = nn.Linear(tensor_dimensions, ff_dimensions)
        self.lt2 = nn.Linear(tensor_dimensions, ff_dimensions)
        self.relu = nn.ReLU()   # relu activation function

    # apply transformations, where x is the input tensor
    def forward(self, x):
        return self.lt2(self.relu(self.lt1(x)))

# positional encoding demarcated by sinusoidal offsets
class PositionalEncoder(nn.Module):
    def __init__(self, tensor_dimensions, max_length):
        super(PositionalEncoder, self).__init__()
        
        pe = torch.zeros(max_length, tensor_dimensions)                 # initialize positional encoding tensor with zeros
        position = torch.arange(0, max_length).unsqueeze(1)             # represents position indices
        divisor = torch.exp(torch.arange(0, tensor_dimensions, 2) *     # scales sine & cosine functions as exponential decay
                            -(math.log(10000.0) / tensor_dimensions))   

        pe[:, 0::2] = torch.sin(position * divisor)     # calculates sine vals for even-indexed, stores in pe
        pe[:, 1::2] = torch.cos(position * divisor)     # calculates cosine vals for odd-indexed positions, stores in pe
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)                  # buffer for se, allows for persistent storage

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]               # adds positional encoding to input tensor
    
class Decoder(nn.Module):
    def __init__(self, tensor_dimensions, num_heads, ff_dimensions, dropout):
        super(Decoder, self).__init__()

        # initialize modules/layers
        print("at decoder layer: ", tensor_dimensions, num_heads)
        self.attn = MaskedMultiHeadAttention(num_heads, tensor_dimensions)
        self.ff = PositionWiseFF(tensor_dimensions, ff_dimensions)
        self.norm1 = nn.LayerNorm(tensor_dimensions)    # 'Norm' part of Add & Norm module
        self.norm2 = nn.LayerNorm(tensor_dimensions)
        self.dropout = nn.Dropout(dropout)        

    # modifies input tensor via: apply self attn, feed forward, add and normalize w' dropout
    def forward(self, x, mask):
        attn_o = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_o))
        ff_o = self.ff(x)
        x = self.norm2(x + self.dropout(ff_o))
        return x 

# full model/engine - pulls everthing together
class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, tensor_dimensions, num_heads, num_layers, ff_dimensions, max_length, dropout):
        super(DecoderTransformer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, tensor_dimensions)                    # embeddes tokens into tensor
        self.positional_encoding = PositionalEncoder(tensor_dimensions, max_length)     # describes relative position of tokens
        
        # creates as many decoder layers as you want - more the better according to who you ask
        self.decoder_layers = nn.ModuleList([Decoder(tensor_dimensions, num_heads, ff_dimensions, dropout)
                                             for _ in range(num_layers)])
        self.linear = nn.Linear(tensor_dimensions, vocab_size)                          # final linear transformation of output 

    def forward(self, tgt):
        # generate mask
        mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        # prevent tokens from attending to future tokens in the same sequence
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt.size(1), tgt.size(1)), diagonal = 1)).bool() 
        mask = mask & nopeak_mask

        tgt_embedded = self.dropout(self.positional_encoding(self.embedding(tgt)))

        decoder_o = tgt_embedded
        for i in self.decoder_layers:
            decoder_o= i(decoder_o, mask)

        # apply final linear layer
        output = self.linear(decoder_o)

        return output