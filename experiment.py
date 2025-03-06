import torch
from torch.utils.data import DataLoader
from datasets.Waymo import WaymoDataset, waymo_collate_fn, create_idx
from model import OccupancyFlowNetwork
from train import train

should_index = False

tfrecord_path = '../data1/waymo_dataset/uncompressed/tf_example/validation'
idx_path = '../data/idxs_validation_bs_1'

if should_index:
    create_idx(tfrecord_path, idx_path)

dataset = WaymoDataset(tfrecord_path, idx_path)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: waymo_collate_fn(x))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

occupancy_flow_net = OccupancyFlowNetwork(road_map_image_size=224, trajectory_feature_dim=9, 
                                          motion_encoder_hidden_dim=512, motion_encoder_seq_len=11,
                                          token_dim=768, embedding_dim=2048, flow_field_hidden_dim=512).to(device)

# road_map, agent_trajectories, unobserved_positions, future_times, target_velocity, target_occupancy_grid = next(iter(dataloader))
# road_map = road_map.to(device)
# agent_trajectories = agent_trajectories.to(device)
# unobserved_positions = unobserved_positions.to(device)
# future_times = future_times.to(device)
# target_velocity = target_velocity.to(device)
# target_occupancy_grid = target_occupancy_grid.to(device)

# print(f'road map: {road_map.shape}')
# print(f'agent trajectories: {agent_trajectories.shape}')
# print(f'unobserved positions: {unobserved_positions.shape}')
# print(f'future times: {future_times.shape}')
# print(f'target velocity: {target_velocity.shape}')
# flow = occupancy_flow_net(future_times, unobserved_positions, road_map, agent_trajectories)
# print(f'flow: {flow.shape}')

train(dataloader, 
      occupancy_flow_net, 
      epochs=1, 
      lr=1e-3,
      weight_decay=0,
      gamma=0.999,
      device=device)