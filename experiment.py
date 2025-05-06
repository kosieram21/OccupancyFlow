import os
import uuid
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from datasets import WaymoCached, waymo_cached_collate_fn
from model import OccupancyFlowNetwork
from train import train
from evaluate import evaluate

@dataclass
class TrainConfig:
    logging_enabled: bool
    checkpointing_enabled: bool
    initialize_from_checkpoint: bool
    should_train: bool
    should_evaluate: bool
    train_path: str
    test_path: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    gamma: float
    road_map_image_size: int 
    road_map_window_size: int 
    trajectory_feature_dim: int
    embedding_dim: int
    flow_field_hidden_dim: int
    flow_field_fourier_features: int

def build_model(config, device):
    model = OccupancyFlowNetwork(
        road_map_image_size=config.road_map_image_size, 
        road_map_window_size=config.road_map_window_size, 
		trajectory_feature_dim=config.trajectory_feature_dim, 
		embedding_dim=config.embedding_dim, 
		flow_field_hidden_dim=config.flow_field_hidden_dim, 
        flow_field_fourier_features=config.flow_field_fourier_features
    ).to(device)

    if config.initialize_from_checkpoint:
        model.load_state_dict(torch.load(f'checkpoints/occupancy_flow_checkpoint{config.epochs - 1}.pt'))

    return model

def prepare_dataset(config, is_train=True, distributed=False, rank=0, world_size=1):
    path = config.train_path if is_train else config.test_path
    dataset = WaymoCached(path)

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, drop_last=is_train)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=min(config.batch_size, torch.get_num_threads()),
        collate_fn=waymo_cached_collate_fn,
        pin_memory=True
    )

    return dataloader

def single_device_train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.logging_enabled:
        experiment_id = str(uuid.uuid4())
        wandb.init(
            project="occupancy-flow",
            name=experiment_id,
            config=config.__dict__
        )
    
    model = build_model(config, device)
    train_dataloader = prepare_dataset(config, is_train=True, distributed=False)
    test_dataloader = prepare_dataset(config, is_train=False, distributed=False)

    train(dataloader=train_dataloader, model=model, device=device, 
          epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay, gamma=config.gamma,
          logging_enabled=config.logging_enabled, checkpointing_enabled=config.checkpointing_enabled)
    
    epe = evaluate(dataloader=test_dataloader, model=model, device=device)

    if config.logging_enabled:
        wandb.log({'epe': epe})
        print(f'end point error: {epe}')

def distributed_train(rank, world_size, config, experiment_id):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if config.logging_enabled:
        wandb.init(
            project='occupancy-flow', 
            name=f'{experiment_id}-{rank}',
            config=config.__dict__,
            mode='online' if rank == 0 else 'disabled'
        )

    try:
        model = build_model(config, rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

        train_dataloader = prepare_dataset(config, is_train=True, distributed=True, rank=rank, world_size=world_size)
        test_dataloader = prepare_dataset(config, is_train=False, distributed=True, rank=rank, world_size=world_size)

        train(dataloader=train_dataloader, model=model, device=rank, 
              epochs=config.epochs, lr=config.lr, weight_decay=config.weight_decay, gamma=config.gamma,
              logging_enabled=config.logging_enabled, checkpointing_enabled=config.checkpointing_enabled)
    
        epe = evaluate(dataloader=test_dataloader, model=model, device=rank)

        if config.logging_enabled and rank==0:
            wandb.log({'epe': epe})
            print(f'end point error: {epe}')
    
    finally:
        dist.barrier()
        dist.destroy_process_group()

def multi_device_train(config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = torch.cuda.device_count()
    experiment_id = str(uuid.uuid4())
    mp.spawn(distributed_train, args=(world_size, config, experiment_id,), nprocs=world_size, join=True)

if __name__ == '__main__':
    data_parallel = False
    
    config = TrainConfig(
        logging_enabled=True,
        checkpointing_enabled=False,
        initialize_from_checkpoint=False,
        should_train=True,
        should_evaluate=True,
        train_path = '../data1/waymo_dataset/v1.1/tensor_cache/training',
        test_path = '../data1/waymo_dataset/v1.1/tensor_cache/validation',
        batch_size=16,
        epochs=100,
        lr=1e-4,
        weight_decay=0,
        gamma=0.999,
        road_map_image_size=256,
        road_map_window_size=8,
        trajectory_feature_dim=10,
        embedding_dim=256,
        flow_field_hidden_dim=256,
        flow_field_fourier_features=128
    )

    if config.logging_enabled:
        wandb.login()

    if torch.cuda.is_available() and data_parallel:
        multi_device_train(config)
    else:
        single_device_train(config)
