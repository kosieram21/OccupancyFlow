import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, token_dim, num_layers, num_heads):
        super().__init__()

        self.token_dim = token_dim
        self.positional_enocding = PositionalEncoding(d_model=token_dim)
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layers, num_layers=num_layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # TODO: should positional encoding and scaling be d_model be in this class?
        # we are going to do intermediate fusion with the visual features so a cross-attention
        # fusion module will be sandwitched between two of these Transformer Modules
        x = x * math.sqrt(self.token_dim)
        x = self.positional_enocding(x)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        embedding = self.encoder(x, mask=causal_mask, is_causal=True)

        if embedding.size(0) == 1:
            embedding = embedding.squeeze(0)

        return embedding