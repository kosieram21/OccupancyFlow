import torch
import torch.nn as nn
from model.layers import CDE
from model.layers import GRU
from model.layers import Transformer

# TODO: need to work on encoder model inputs
class Encoder(nn.Module):
    def __init__(self, 
                 trajectory_feature_dim, token_dim, embedding_dim,
                 motion_encoder_hidden_dim, motion_encoder_seq_len):
        super(Encoder, self).__init__()

        assert embedding_dim % 2 == 0, "embedding_dim must be divisible by 2 for bidirectional GRU"

        self.motion_encoder_seq_len = motion_encoder_seq_len
        self.motion_encoder = CDE(input_dim=trajectory_feature_dim, 
                                  embedding_dim=token_dim, 
                                  hidden_dim=motion_encoder_hidden_dim, 
                                  num_layers=4)
        
        self.visual_encoder = None
        
        self.self_attention_transformer = Transformer(token_dim=token_dim,
                                                      num_layers=4,
                                                      num_heads=8)
        
        self.pooling_module = GRU(input_dim=token_dim,
                                  hidden_dim=embedding_dim // 2,
                                  num_layers=4,
                                  bidirectional=True)

    def forward(self, agent_trajectories, road_graph, traffic_light_state):
        t = torch.linspace(0., 1., self.motion_encoder_seq_len).to(agent_trajectories)
        print(agent_trajectories.shape)
        print(t.shape)
        tokens = self.motion_encoder(t, agent_trajectories)
        print(tokens.shape)
        tokens = self.self_attention_transformer(tokens)
        # self-attention transformer
        # cross-attention (road_graph) transformer
        # self-attention transformer
        # cross-attention (traffic_light_state) transformer
        # self-attention transformer
        print(tokens.shape)
        embedding = self.pooling_module(tokens)
        print(embedding.shape)
        return embedding