import torch.nn as nn
from model.layers import GRU
from model.layers import SelfAttentionTransformer
from model.layers import CrossAttentionTransformer
from model.layers import SwinTransformer

class SceneEncoder(nn.Module):
    def __init__(self, 
                 road_map_image_size, road_map_window_size, 
                 trajectory_feature_dim, 
                 embedding_dim):
        super(SceneEncoder, self).__init__()

        assert road_map_image_size % road_map_window_size == 0, "road_map_image_size must be divisible by road_map_window_size"
        assert embedding_dim % 2 == 0, "embedding_dim must be divisible by 2 for bidirectional GRU"
        assert embedding_dim % 8 == 0, "tokken_dim must be divisible by 8 for swin-t patch-merging"

        self.motion_encoder = GRU(input_dim=trajectory_feature_dim,
                                  hidden_dim=embedding_dim // 2,
                                  bidirectional=True)
        
        self.visual_encoder = SwinTransformer(img_size=road_map_image_size,
                                              embed_dim=embedding_dim // 8,
                                              window_size=road_map_window_size,
                                              num_heads=[8,8,8,8])
        
        self.interaction_transformer1 = SelfAttentionTransformer(token_dim=embedding_dim,
                                                                 num_layers=4,
                                                                 num_heads=8,
                                                                 mlp_dim=4 * embedding_dim)
        
        self.fusion_transformer = CrossAttentionTransformer(token_dim=embedding_dim,
                                                            num_layers=4,
                                                            num_heads=8,
                                                            mlp_dim=4 * embedding_dim)
        
        self.interaction_transformer2 = SelfAttentionTransformer(token_dim=embedding_dim,
                                                                 num_layers=4,
                                                                 num_heads=8,
                                                                 mlp_dim=4 * embedding_dim)

    def forward(self, road_map, agent_trajectories, agent_mask=None):
        agent_tokens = self.motion_encoder(agent_trajectories, agent_mask)
        environment_tokens = self.visual_encoder(road_map)
        agent_tokens = self.interaction_transformer1(agent_tokens, agent_mask)
        agent_tokens = agent_tokens + self.fusion_transformer(agent_tokens, environment_tokens, agent_mask)
        agent_tokens = self.interaction_transformer2(agent_tokens, agent_mask)
        agent_tokens = agent_tokens * agent_mask.unsqueeze(-1) if agent_mask is not None else agent_tokens
        agents_per_scene = agent_mask.sum(dim=-1)
        embedding = agent_tokens.sum(dim=1) / agents_per_scene.unsqueeze(-1)
        return embedding