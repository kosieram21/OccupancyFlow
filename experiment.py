import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.Waymo import WaymoDataset, waymo_collate_fn, create_idx
from model import OccupancyFlowNetwork
from train import train

should_index = False

tfrecord_path = '../data1/waymo_dataset/uncompressed/tf_example/validation'
idx_path = '../idx/validation'

if should_index:
	create_idx(tfrecord_path, idx_path)

batch_size = 6
#batch_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
dataset = WaymoDataset(tfrecord_path, idx_path)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: waymo_collate_fn(x))

occupancy_flow_net = OccupancyFlowNetwork(road_map_image_size=224, trajectory_feature_dim=10, 
										  motion_encoder_hidden_dim=512, motion_encoder_seq_len=11,
										  flow_field_hidden_dim=512, flow_field_fourier_features=128,
										  token_dim=768, embedding_dim=128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Detected Device: {device}')

#if torch.cuda.device_count() > 1: # TODO: need to be able to produce batches larger than 1 to use nn.DataParallel
#	print(f'Using data parallelism across {torch.cuda.device_count()} GPUs')
#	occupancy_flow_net = nn.DataParallel(occupancy_flow_net)

occupancy_flow_net = occupancy_flow_net.to(device)

road_map, agent_trajectories, \
unobserved_positions, future_times, target_velocity, \
agent_trajectory_mask, flow_field_mask = next(iter(dataloader))

road_map = road_map.to(device)
agent_trajectories = agent_trajectories.to(device)


import time

start = time.time()
embedding = occupancy_flow_net.scence_encoder(road_map, agent_trajectories, agent_trajectory_mask)
end = time.time()
elapsed = end - start
print(elapsed)

#train(dataloader, 
#	  occupancy_flow_net, 
#	  epochs=10, 
#	  lr=1e-3,
#	  weight_decay=0,
#	  gamma=0.999,
#	  device=device)