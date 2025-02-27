import torch
import torch.nn as nn
from model.layers.CDE import CDE

class Encoder(nn.Module):
    def __init__(self, 
                 trajectory_feature_dim, trajectory_embedding_dim, 
                 motion_encoder_hidden_dim, motion_encoder_seq_len):
        super(Encoder, self).__init__()
        self.motion_encoder = CDE(input_dim=trajectory_feature_dim, 
                                  embedding_dim=trajectory_embedding_dim, 
                                  hidden_dim=motion_encoder_hidden_dim, 
                                  num_layers=4)
        self.motion_encoder_seq_len = motion_encoder_seq_len

    def forward(self, agent_trajectories, road_graph, traffic_light_state):
        t = torch.linspace(0., 1., self.motion_encoder_seq_len).to(agent_trajectories)
        tokens = self.motion_encoder(t, agent_trajectories)
        # self-attention transformer
        # cross-attention (road_graph) transformer
        # self-attention transformer
        # cross-attention (traffic_light_state) transformer
        # self-attention transformer
        # pooling-module (GRU?)