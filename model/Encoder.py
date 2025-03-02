import math
import torch
import torch.nn as nn
from model.layers import CDE
from model.layers import GRU
from model.layers import SelfAttentionTransformer
from model.layers import SwinTransformer

# TODO: need to work on encoder model inputs
class Encoder(nn.Module):
    def __init__(self, 
                 road_map_image_size, trajectory_feature_dim, 
                 motion_encoder_hidden_dim, motion_encoder_seq_len,
                 token_dim, embedding_dim):
        super(Encoder, self).__init__()

        assert embedding_dim % 2 == 0, "embedding_dim must be divisible by 2 for bidirectional GRU"

        self.motion_encoder_seq_len = motion_encoder_seq_len
        self.motion_encoder = CDE(input_dim=trajectory_feature_dim, 
                                  embedding_dim=token_dim, 
                                  hidden_dim=motion_encoder_hidden_dim, 
                                  num_layers=4)
        
        # TODO: need to figure out how to properly configure the Swin-T
        self.visual_encoder = SwinTransformer(img_size=road_map_image_size,
                                              embed_dim=96)
        
        self.self_attention_transformer = SelfAttentionTransformer(token_dim=token_dim,
                                                                   num_layers=4,
                                                                   num_heads=8)
        
        # TODO: What is the appropriate pooling module
        self.pooling_module = GRU(input_dim=token_dim,
                                  hidden_dim=embedding_dim // 2,
                                  num_layers=4,
                                  bidirectional=True)

    def forward(self, road_map, agent_trajectories):
        t = torch.linspace(0., 1., self.motion_encoder_seq_len).to(agent_trajectories)
        print(agent_trajectories.shape)
        print(t.shape)
        agent_tokens = self.motion_encoder(t, agent_trajectories)
        print(agent_tokens.shape)
        print(road_map.shape)
        environment_tokens = self.visual_encoder(road_map)
        print(environment_tokens.shape)
        agent_tokens = self.self_attention_transformer(agent_tokens)
        # cross-attention (rasterized road map) transformer
        # self-attention transformer
        print(agent_tokens.shape)
        embedding = self.pooling_module(agent_tokens)
        print(embedding.shape)
        return embedding