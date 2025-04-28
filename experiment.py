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
    tfrecord_path: str
    idx_path: str
    batch_size: int
    batches_per_epoch: int
    epochs: int
    lr: float
    weight_decay: float
    gamma: float
    road_map_image_size: int
    trajectory_feature_dim: int
    motion_encoder_hidden_dim: int
    motion_encoder_seq_len: int
    visual_encoder_hidden_dim: int 
    visual_encoder_window_size: int
    flow_field_hidden_dim: int
    flow_field_fourier_features: int
    token_dim: int
    embedding_dim: int

def single_device_train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_id = str(uuid.uuid4())
    wandb.init(
        project="occupancy-flow",
        name=experiment_id,
        config=config.__dict__
    )

    dataset = WaymoCached('../data1/waymo_dataset/v1.1/tensor_cache/training')
    dataloader = DataLoader(dataset, 
                            batch_size=config.batch_size, 
                            collate_fn=waymo_cached_collate_fn,
                            pin_memory=True)

    model = OccupancyFlowNetwork(
        road_map_image_size=config.road_map_image_size,
        trajectory_feature_dim=config.trajectory_feature_dim,
        motion_encoder_hidden_dim=config.motion_encoder_hidden_dim,
        motion_encoder_seq_len=config.motion_encoder_seq_len, 
        visual_encoder_hidden_dim=config.visual_encoder_hidden_dim,
        visual_encoder_window_size=config.visual_encoder_window_size,                            
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
        device=device,
        logging_enabled=config.logging_enabled,
        checkpointing_enabled=config.checkpointing_enabled
    )

def distributed_train(rank, world_size, config, experiment_id):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # wandb.init(
    #     project="occupancy-flow", 
    #     name=f"{experiment_id}-{rank}",
    #     config=config.__dict__,
    #     mode="online" if rank == 0 else "disabled"
    # )

    try:
        # train_dataset = WaymoCached('../data1/waymo_dataset/v1.1/tensor_cache/training')
        # train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        # train_dataloader = DataLoader(dataset, 
        #                         batch_size=config.batch_size, 
        #                         sampler=sampler, 
        #                         num_workers=min(config.batch_size, torch.get_num_threads()), 
        #                         collate_fn=waymo_cached_collate_fn,
        #                         pin_memory=True)

        model = OccupancyFlowNetwork(
            road_map_image_size=config.road_map_image_size,
            trajectory_feature_dim=config.trajectory_feature_dim,
            motion_encoder_hidden_dim=config.motion_encoder_hidden_dim,
            motion_encoder_seq_len=config.motion_encoder_seq_len, 
            visual_encoder_hidden_dim=config.visual_encoder_hidden_dim,
            visual_encoder_window_size=config.visual_encoder_window_size,                         
            flow_field_hidden_dim=config.flow_field_hidden_dim,
            flow_field_fourier_features=config.flow_field_fourier_features,
            token_dim=config.token_dim,
            embedding_dim=config.embedding_dim
        ).to(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        model.load_state_dict(torch.load('checkpoints/occupancy_flow_checkpoint99.pt')) # should have a bool inidcating if we should load model params
    
        # train(
        #     dataloader=train_dataloader, 
        #     model=model, 
        #     epochs=config.epochs, 
        #     lr=config.lr, 
        #     weight_decay=config.weight_decay, 
        #     gamma=config.gamma, 
        #     device=rank,
        #     logging_enabled=config.logging_enabled and rank==0,
        #     checkpointing_enabled=config.checkpointing_enabled
        # )

        test_dataset = WaymoCached('../data1/waymo_dataset/v1.1/tensor_cache/validation')
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, 
                                batch_size=config.batch_size, 
                                sampler=test_sampler, 
                                num_workers=min(config.batch_size, torch.get_num_threads()), 
                                collate_fn=waymo_cached_collate_fn,
                                pin_memory=True)
        
        epe = evaluate(
            dataloader=test_dataloader,
            model=model,
            device=rank
        )
        print(f'test set end point error: {epe}')
    
    finally:
        dist.barrier()
        dist.destroy_process_group()

def multi_device_train(config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = torch.cuda.device_count()
    experiment_id = str(uuid.uuid4())
    mp.spawn(distributed_train, args=(world_size, config, experiment_id,), nprocs=world_size, join=True)

if __name__ == "__main__":
    data_parallel = True
    
    config = TrainConfig(
        logging_enabled=True,
        checkpointing_enabled=True,
        tfrecord_path='../data1/waymo_dataset/v1.1/waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/training',
        idx_path='../data1/waymo_dataset/v1.1/idx/training',
        batch_size=16,
        batches_per_epoch=1600,#2000,
        epochs=100,
        lr=1e-4,
        weight_decay=0,
        gamma=0.999,
        road_map_image_size=256,
        trajectory_feature_dim=10,
        motion_encoder_hidden_dim=256,
        motion_encoder_seq_len=11,
        visual_encoder_hidden_dim=256,
        visual_encoder_window_size=8,
        flow_field_hidden_dim=256,
        flow_field_fourier_features=128,
        token_dim=256,
        embedding_dim=256
    )

    #wandb.login()

    if torch.cuda.is_available() and data_parallel:
        multi_device_train(config)
    else:
        single_device_train(config)
