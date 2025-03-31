import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets.Waymo import WaymoDataset, waymo_collate_fn, create_idx
from model import OccupancyFlowNetwork
from train import train

def single_device_train(tfrecord_path, idx_path, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = WaymoDataset(tfrecord_path, idx_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: waymo_collate_fn(x))

    occupancy_flow_net = OccupancyFlowNetwork(road_map_image_size=224, trajectory_feature_dim=10, 
                                              motion_encoder_hidden_dim=512, motion_encoder_seq_len=11,
                                              flow_field_hidden_dim=512, flow_field_fourier_features=128,
                                              token_dim=768, embedding_dim=128)
    #occupancy_flow_net = torch.compile(occupancy_flow_net) #TODO: can we get compile to work and be fast?
    occupancy_flow_net = occupancy_flow_net.to(device)

    train(dataloader, occupancy_flow_net, epochs=10, lr=1e-3, weight_decay=0, gamma=0.999, device=device)

def distributed_train(rank, world_size, tfrecord_path, idx_path, batch_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    #sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) 
    # TODO: are we even splitting the data correctly without the sampler??
    dataset = WaymoDataset(tfrecord_path, idx_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: waymo_collate_fn(x))
    
    occupancy_flow_net = OccupancyFlowNetwork(road_map_image_size=224, trajectory_feature_dim=10, 
                                              motion_encoder_hidden_dim=512, motion_encoder_seq_len=11,
                                              flow_field_hidden_dim=512, flow_field_fourier_features=128,
                                              token_dim=768, embedding_dim=128)
    #occupancy_flow_net = torch.compile(occupancy_flow_net) #TODO: can we get compile to work and be fast?
    occupancy_flow_net.to(rank)
    occupancy_flow_net = nn.parallel.DistributedDataParallel(occupancy_flow_net, 
                                                             device_ids=[rank], 
                                                             find_unused_parameters=True)
    
    train(dataloader, occupancy_flow_net, epochs=10, lr=1e-3, weight_decay=0, gamma=0.999, device=rank)
    
    dist.destroy_process_group()

def multi_device_train(tfrecord_path, idx_path, batch_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    world_size = torch.cuda.device_count()
    mp.spawn(distributed_train, args=(world_size, tfrecord_path, idx_path, batch_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    should_index = False
    data_parallel = True

    tfrecord_path = '../data1/waymo_dataset/uncompressed/tf_example/validation'
    idx_path = '../idx/validation'
    batch_size = 16

    if should_index:
        create_idx(tfrecord_path, idx_path)

    if torch.cuda.is_available() and data_parallel:
        multi_device_train(tfrecord_path, idx_path, batch_size)
    else:
        single_device_train(tfrecord_path, idx_path, batch_size)
