# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from datasets.Waymo import WaymoDataset, waymo_collate_fn, create_idx
# from model import OccupancyFlowNetwork
# from train import train

# should_index = False
# data_parallel = True

# tfrecord_path = '../data1/waymo_dataset/uncompressed/tf_example/validation'
# idx_path = '../idx/validation'

# if should_index:
# 	create_idx(tfrecord_path, idx_path)

# # EFFICENCY NOTES:
# #   1. Spline/CDE is slow causing no efficency benifit with batching (without Spline/CDE batching improves efficency)
# #   2. nn.DataParallel makes traing twice as slow instead of 4 times as fast (as would be expected on a system with 4 GPUs)

# PER_DEVICE_BATCH_SIZE = 16
# batch_size = PER_DEVICE_BATCH_SIZE * (torch.cuda.device_count() if data_parallel and torch.cuda.is_available() else 1)
# dataset = WaymoDataset(tfrecord_path, idx_path)
# dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: waymo_collate_fn(x))

# occupancy_flow_net = OccupancyFlowNetwork(road_map_image_size=224, trajectory_feature_dim=10, 
# 										  motion_encoder_hidden_dim=512, motion_encoder_seq_len=11,
# 										  flow_field_hidden_dim=512, flow_field_fourier_features=128,
# 										  token_dim=768, embedding_dim=128)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Detected Device: {device}')

# if data_parallel and torch.cuda.device_count() > 1:
# 	print(f'Using data parallelism across {torch.cuda.device_count()} GPUs')
# 	occupancy_flow_net = nn.DataParallel(occupancy_flow_net)

# occupancy_flow_net = occupancy_flow_net.to(device)

# train(dataloader, 
# 	  occupancy_flow_net, 
# 	  epochs=10, 
# 	  lr=1e-3,
# 	  weight_decay=0,
# 	  gamma=0.999,
# 	  device=device)

# TODO: This is a mess but can we get DDP working?

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from datasets.Waymo import WaymoDataset, waymo_collate_fn, create_idx
from model import OccupancyFlowNetwork
from train import train

should_index = False
data_parallel = True

tfrecord_path = '../data1/waymo_dataset/uncompressed/tf_example/validation'
idx_path = '../idx/validation'

if should_index:
    create_idx(tfrecord_path, idx_path)

PER_DEVICE_BATCH_SIZE = 16
batch_size = 16#PER_DEVICE_BATCH_SIZE * (torch.cuda.device_count() if data_parallel and torch.cuda.is_available() else 1)
dataset = WaymoDataset(tfrecord_path, idx_path)

# Set the environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'  # Since it's a single-node setup, use localhost
os.environ['MASTER_PORT'] = '12355'  # You can choose any open port, just ensure it's not in use by other processes
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# Create the dataloader without any changes; it will be passed with DistributedSampler later
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: waymo_collate_fn(x))

occupancy_flow_net = OccupancyFlowNetwork(road_map_image_size=224, trajectory_feature_dim=10, 
                                          motion_encoder_hidden_dim=512, motion_encoder_seq_len=11,
                                          flow_field_hidden_dim=512, flow_field_fourier_features=128,
                                          token_dim=768, embedding_dim=128)

# Automatically detect available device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Detected Device: {device}')

# Set up the training with DDP (Distributed Data Parallel)
def main(rank, world_size):
    # Choose the backend based on device availability
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    
    # Initialize the distributed process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Set device for each process (GPU or CPU)
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None

    # Create DistributedSampler for the dataset and update the DataLoader
    #sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) 
    # TODO: are we even splitting the data correctly without the sampler??
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: waymo_collate_fn(x))
    
    # Move model to the correct device (GPU or CPU)
    occupancy_flow_net.to(device)
    
    # Wrap the model with DDP
    model = nn.parallel.DistributedDataParallel(occupancy_flow_net, device_ids=[rank] if torch.cuda.is_available() else None, find_unused_parameters=True)
    
    # Train the model
    train(dataloader, model, epochs=10, lr=1e-3, weight_decay=0, gamma=0.999, device=rank)
    
    # Clean up distributed environment after training
    dist.destroy_process_group()

def setup():
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1  # Set world_size based on available GPUs
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    setup()
