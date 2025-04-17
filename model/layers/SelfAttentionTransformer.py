import torch.nn as nn

class SelfAttentionTransformer(nn.Module):
    def __init__(self, token_dim, num_layers, num_heads, mlp_dim):
        super().__init__()
        
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layers, num_layers=num_layers)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = (mask == 0)
        x = self.encoder(x, src_key_padding_mask=mask, is_causal=False)
        return x