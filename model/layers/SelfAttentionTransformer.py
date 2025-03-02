import torch.nn as nn

class SelfAttentionTransformer(nn.Module):
    def __init__(self, token_dim, num_layers, num_heads):
        super().__init__()
        self.d_model = token_dim
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layers, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x, is_causal=False)
        return x