import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from dataclasses import dataclass
from datasets.Waymo import WaymoDataset, waymo_collate_fn, create_idx
from model import OccupancyFlowNetwork
from train import train

@dataclass
class TrainConfig:
    tfrecord_path: str
    idx_path: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    gamma: float
    road_map_image_size: int
    trajectory_feature_dim: int
    motion_encoder_hidden_dim: int
    motion_encoder_seq_len: int
    flow_field_hidden_dim: int
    flow_field_fourier_features: int
    token_dim: int
    embedding_dim: int

def single_device_train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = WaymoDataset(config.tfrecord_path, config.idx_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=lambda x: waymo_collate_fn(x))

    model = OccupancyFlowNetwork(
        road_map_image_size=config.road_map_image_size, 
        trajectory_feature_dim=config.trajectory_feature_dim, 
        motion_encoder_hidden_dim=config.motion_encoder_hidden_dim, 
        motion_encoder_seq_len=config.motion_encoder_seq_len,                                      
        flow_field_hidden_dim=config.flow_field_hidden_dim, 
        flow_field_fourier_features=config.flow_field_fourier_features,
        token_dim=config.token_dim, 
        embedding_dim=config.embedding_dim
    ).to(device)

    train(
        dataloader=dataloader, 
        model=model, 
        epochs=config.epochs, 
        lr=config.lr, 
        weight_decay=config.weight_decay, 
        gamma=config.gamma, 
        device=device
    )

def distributed_train(rank, world_size, config):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    dataset = WaymoDataset(config.tfrecord_path, config.idx_path, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=lambda x: waymo_collate_fn(x))
    
    model = OccupancyFlowNetwork(
        road_map_image_size=config.road_map_image_size, 
        trajectory_feature_dim=config.trajectory_feature_dim, 
        motion_encoder_hidden_dim=config.motion_encoder_hidden_dim, 
        motion_encoder_seq_len=config.motion_encoder_seq_len,                                      
        flow_field_hidden_dim=config.flow_field_hidden_dim, 
        flow_field_fourier_features=config.flow_field_fourier_features,
        token_dim=config.token_dim, 
        embedding_dim=config.embedding_dim
    ).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    
    train(
        dataloader=dataloader, 
        model=model, 
        epochs=config.epochs, 
        lr=config.lr, 
        weight_decay=config.weight_decay, 
        gamma=config.gamma, 
        device=rank
    )
    
    dist.destroy_process_group()

def multi_device_train(config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    world_size = torch.cuda.device_count()
    mp.spawn(distributed_train, args=(world_size, config,), nprocs=world_size, join=True)

if __name__ == "__main__":
    should_index = False
    data_parallel = True
    
    config = TrainConfig(
        tfrecord_path='../data1/waymo_dataset/uncompressed/tf_example/validation',
        idx_path='../idx/validation',
        batch_size=16,
        epochs=10,
        lr=1e-3,
        weight_decay=0,
        gamma=0.999,
        road_map_image_size=224,
        trajectory_feature_dim=10,
        motion_encoder_hidden_dim=512,
        motion_encoder_seq_len=11,
        flow_field_hidden_dim=512,
        flow_field_fourier_features=128,
        token_dim=768,
        embedding_dim=128
    )

    if should_index:
        create_idx(config.tfrecord_path, config.idx_path)

    if torch.cuda.is_available() and data_parallel:
        multi_device_train(config)
    else:
        single_device_train(config)
