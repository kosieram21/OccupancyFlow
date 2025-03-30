import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.Waymo import WaymoDataset, waymo_collate_fn, create_idx
from model import OccupancyFlowNetwork
from train import train

should_index = False
data_parallel = True

tfrecord_path = '../data1/waymo_dataset/uncompressed/tf_example/validation'
idx_path = '../idx/validation'

if should_index:
	create_idx(tfrecord_path, idx_path)

# EFFICENCY NOTES:
#   1. Spline/CDE is slow causing no efficency benifit with batching (without Spline/CDE batching improves efficency)
#   2. nn.DataParallel makes traing twice as slow instead of 4 times as fast (as would be expected on a system with 4 GPUs)

PER_DEVICE_BATCH_SIZE = 16
batch_size = PER_DEVICE_BATCH_SIZE * (torch.cuda.device_count() if data_parallel and torch.cuda.is_available() else 1)
dataset = WaymoDataset(tfrecord_path, idx_path)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: waymo_collate_fn(x))

occupancy_flow_net = OccupancyFlowNetwork(road_map_image_size=224, trajectory_feature_dim=10, 
										  motion_encoder_hidden_dim=512, motion_encoder_seq_len=11,
										  flow_field_hidden_dim=512, flow_field_fourier_features=128,
										  token_dim=768, embedding_dim=128)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Detected Device: {device}')

if data_parallel and torch.cuda.device_count() > 1:
	print(f'Using data parallelism across {torch.cuda.device_count()} GPUs')
	occupancy_flow_net = nn.DataParallel(occupancy_flow_net)

occupancy_flow_net = occupancy_flow_net.to(device)

train(dataloader, 
	  occupancy_flow_net, 
	  epochs=10, 
	  lr=1e-3,
	  weight_decay=0,
	  gamma=0.999,
	  device=device)