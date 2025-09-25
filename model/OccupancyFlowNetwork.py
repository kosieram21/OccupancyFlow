import torch
import torch.nn as nn
from model.layers.ODE import ODE
from model.layers.ConditionedMLP import ConditionedMLP
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
		
		# TODO: we are post training these guys. Should the be trained during the pre-train?
		self.occupancy_estimation_head = ConditionedMLP(3, embedding_dim, 
													   (flow_field_hidden_dim for _ in range(4)), 
													    2, flow_field_fourier_features)

	def forward(self, t, h, road_map, agent_trajectories, agent_mask=None):
		scene_context = self.scene_encoder(road_map, agent_trajectories, agent_mask)
		flow = self.flow_field(t, h, scene_context)
		return flow, scene_context
	
	def warp_occupancy(self, occupancy, integration_times, road_map, agent_trajectories, agent_mask=None):
		scene_context = self.scene_encoder(road_map, agent_trajectories, agent_mask)
		estimated_occupancy = self.flow_field.solve_ivp(occupancy, integration_times, scene_context)
		return estimated_occupancy, scene_context
	
	def estimate_occupancy(self, t, h, road_map, agent_trajectories, agent_mask=None):
		spatiotemporal_coordinates = torch.cat([h, t], dim=-1)
		scene_context = self.scene_encoder(road_map, agent_trajectories, agent_mask)
		occupancy = self.occupancy_estimation_head(spatiotemporal_coordinates, scene_context)
		return occupancy, scene_context
	