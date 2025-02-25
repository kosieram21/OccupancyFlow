import torch
import torch.nn as nn
from model.layers.CDE import CDE

class Encoder(nn.Module):
    def __init__(self, trajectory_feature_dim, trajectory_embedding_dim, motion_encoder_hidden_dim):
        self.motion_encoder = CDE(input_dim=trajectory_feature_dim, 
                                  embedding_dim=trajectory_embedding_dim, 
                                  hidden_dim=motion_encoder_hidden_dim, 
                                  num_layers=4)

    def forward(self, agent_trajectories, road_graph, traffic_light_state):
        # time t needs to go into the model...
        # probably should not be the time from the dataset and instead a normalized time (1/11th)
        tokens = self.motion_encoder(agent_trajectories)