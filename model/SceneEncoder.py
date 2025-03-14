import math
import torch
import torch.nn as nn
from model.layers import CDE
from model.layers import GRU
from model.layers import SelfAttentionTransformer
from model.layers import CrossAttentionTransformer
from model.layers import SwinTransformer

# TODO: need to work on encoder model inputs
class SceneEncoder(nn.Module):
    def __init__(self, 
                 road_map_image_size, trajectory_feature_dim, 
                 motion_encoder_hidden_dim, motion_encoder_seq_len,
                 token_dim, embedding_dim):
        super(SceneEncoder, self).__init__()

        assert embedding_dim % 2 == 0, "embedding_dim must be divisible by 2 for bidirectional GRU"

        self.motion_encoder_seq_len = motion_encoder_seq_len
        self.motion_encoder = CDE(input_dim=trajectory_feature_dim, 
                                  embedding_dim=token_dim, 
                                  hidden_dim=motion_encoder_hidden_dim, 
                                  num_layers=4)
        
        # TODO: need to figure out how to properly configure the Swin-T
        self.visual_encoder = SwinTransformer(img_size=road_map_image_size,
                                              embed_dim=96)
        
        self.interaction_transformer1 = SelfAttentionTransformer(token_dim=token_dim,
                                                                 num_layers=4,
                                                                 num_heads=8)
        
        self.fusion_transformer = CrossAttentionTransformer(token_dim=token_dim,
                                                            num_layers=4,
                                                            num_heads=8)
        
        self.interaction_transformer2 = SelfAttentionTransformer(token_dim=token_dim,
                                                                 num_layers=4,
                                                                 num_heads=8)
        
        # TODO: What is the appropriate pooling module
        self.pooling_module = GRU(input_dim=token_dim,
                                  hidden_dim=embedding_dim // 2,
                                  num_layers=4,
                                  bidirectional=True)

    def forward(self, road_map, agent_trajectories):
        t = torch.linspace(0., 1., self.motion_encoder_seq_len).to(agent_trajectories)
        agent_tokens = self.motion_encoder(t, agent_trajectories)
        #environment_tokens = self.visual_encoder(road_map)
        #agent_tokens = self.interaction_transformer1(agent_tokens)
        #agent_tokens = agent_tokens + self.fusion_transformer(agent_tokens, environment_tokens)
        #agent_tokens = self.interaction_transformer2(agent_tokens)
        embedding = self.pooling_module(agent_tokens)
        return embedding