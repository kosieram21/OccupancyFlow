import torch # TODO: delete me
import torch.nn as nn
from model.layers.ODE import ODE
from model.SceneEncoder import SceneEncoder

class OccupancyFlowNetwork(nn.Module):
	def __init__(self, 
			  	 road_map_image_size, road_map_window_size, 
				 trajectory_feature_dim, 
				 embedding_dim, 
				 flow_field_hidden_dim, flow_field_fourier_features):
		super(OccupancyFlowNetwork, self).__init__()

		self.scene_encoder = SceneEncoder(road_map_image_size, road_map_window_size, 
									 	   trajectory_feature_dim, 
				 						   embedding_dim)
			
		self.flow_field = ODE(2, embedding_dim, 
							 (flow_field_hidden_dim for _ in range(4)), 
							  flow_field_fourier_features)

	def forward(self, t, h, road_map, agent_trajectories, agent_mask=None, flow_field_mask=None):
		scene_context = self.scene_encoder(road_map, agent_trajectories, agent_mask)
		flow = self.flow_field(t, h, scene_context, flow_field_mask)
		return flow
	
	def warp_occupancy(self, occupancy, integration_times, scene_context, use_custom=False):
		if use_custom:
			estimated_occupancy, _ = self.flow_field.solve_ivp2(occupancy, integration_times, scene_context)
		else:
			estimated_occupancy, _ = self.flow_field.solve_ivp(occupancy, integration_times, scene_context)
		estimated_occupancy = [estimated_occupancy[i] for i in range(estimated_occupancy.shape[0])]
		return estimated_occupancy