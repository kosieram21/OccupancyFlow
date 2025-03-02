import math
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, token_dim, num_layers, num_heads):
        super().__init__()
        self.d_model = token_dim
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layers, num_layers=num_layers)

    def forward(self, x):
        # TODO: we should rework the model so that it is batched
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x * math.sqrt(self.d_model) # TODO: Should we do this in every block?
        x = self.encoder(x, is_causal=False)

        if x.size(0) == 1:
            x = x.squeeze(0)

        return x