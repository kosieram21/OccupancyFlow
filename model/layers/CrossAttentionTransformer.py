import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim=2048):
        super().__init__()
        
        self.norm_query = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model)
        )
        
    def forward(self, query, key, mask=None):
        norm_query = self.norm_query(query)
        norm_query = norm_query * mask.unsqueeze(-1) if mask is not None else norm_query # TODO: double check this is correct
        attn_output, _ = self.cross_attention(
            query=norm_query,
            key=key,
            value=key)
        query = query + attn_output
        
        norm_query = self.norm_mlp(query)
        mlp_output = self.mlp(norm_query)
        query = query + mlp_output
        
        return query

class CrossAttentionTransformer(nn.Module):
    def __init__(self, token_dim, num_layers, num_heads, mlp_dim):
        super().__init__()

        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=token_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim
            ) for _ in range(num_layers)
        ])

    def forward(self, x, y, mask=None):
        for layer in self.layers:
            x = layer(query=x, key=y, mask=mask)
        return x