import torch
import torch.nn as nn
from model.layers.ODE import ODE
from model.SceneEncoder import SceneEncoder

class OccupancyFlowNetwork(nn.Module):
	def __init__(self,
				 road_map_image_size, trajectory_feature_dim, 
				 motion_encoder_hidden_dim, motion_encoder_seq_len,
				 token_dim, embedding_dim, flow_field_hidden_dim):
		super(OccupancyFlowNetwork, self).__init__()

		self.scence_encoder = SceneEncoder(road_map_image_size, trajectory_feature_dim, 
										   motion_encoder_hidden_dim, motion_encoder_seq_len,
										   token_dim, embedding_dim)
		self.flow_field = ODE(2, embedding_dim, (flow_field_hidden_dim for _ in range(4)))

	def forward(self, t, h, road_map, agent_trajectories):
		scene_context = self.scence_encoder(road_map, agent_trajectories)
		flow = self.flow_field(t, h, scene_context)
		return flow
	
	def warp_occupancy(self, occupancy, scene_context):
		return self.flow_field.solve_ivp(occupancy, scene_context)