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
from train import pre_train, fine_tune
from evaluate import evaluate
from visualize import visualize

@dataclass
class ExperimentConfig:
    data_parallel: bool
    logging_enabled: bool
    checkpointing_enabled: bool
    initialize_from_checkpoint: bool
    should_pre_train: bool
    should_fine_tune: bool
    should_evaluate: bool
    should_visualize: bool
    pre_train_path: str
    fine_tune_path: str
    test_path: str
    visualization_path: str
    pre_train_batch_size: int
    pre_train_epochs: int
    pre_train_lr: float
    pre_train_weight_decay: float
    pre_train_gamma: float
    fine_tune_batch_size: int
    fine_tune_epochs: int
    fine_tune_lr: float
    fine_tune_weight_decay: float
    fine_tune_gamma: float
    test_batch_size: int
    visualization_batch_size: int
    visualization_samples: int
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
        # TODO: configurable checkpoint root and id
        #model.load_state_dict(torch.load(f'checkpoints/pretrain/occupancy_flow_checkpoint{config.pre_train_epochs - 1}.pt'))
        model.load_state_dict(torch.load(f'checkpoints/finetune/occupancy_flow_checkpoint12.pt'))
        #model.load_state_dict(torch.load(f'checkpoints/pretrain/occupancy_flow_checkpoint99.pt'))
        # TODO: delete the alternative model loading logic
        #checkpoint = 1
        #state_dict = torch.load(f'checkpoints/occupancy_flow_checkpoint{checkpoint}.pt')
        #corrected_state_dict = {k.replace("scence_encoder", "scene_encoder"): v for k, v in state_dict.items()} # TODO: delete me
        #model.load_state_dict(corrected_state_dict)

    return model

def prepare_dataset(path, batch_size, is_train=True, distributed=False, rank=0, world_size=1):
    dataset = WaymoCached(path)

    shuffle=is_train

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=is_train)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=min(batch_size, torch.get_num_threads()),
        collate_fn=waymo_cached_collate_fn,
        pin_memory=True
    )

    return dataloader

def single_device_experiment(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.logging_enabled:
        experiment_id = str(uuid.uuid4())
        wandb.init(
            project="occupancy-flow",
            name=experiment_id,
            config=config.__dict__
        )
    
    model = build_model(config, device)

    if config.should_pre_train:
        pre_train_dataloader = prepare_dataset(config.pre_train_path, config.pre_train_batch_size, is_train=True, distributed=False)
        pre_train(dataloader=pre_train_dataloader, model=model, device=device, 
                  epochs=config.pre_train_epochs, lr=config.pre_train_lr, weight_decay=config.pre_train_weight_decay, gamma=config.pre_train_gamma,
                  logging_enabled=config.logging_enabled, checkpointing_enabled=config.checkpointing_enabled)
    
    if config.should_fine_tune:
        fine_tune_dataloader = prepare_dataset(config.fine_tune_path, config.fine_tune_batch_size, is_train=True, distributed=False)
        fine_tune(dataloader=fine_tune_dataloader, model=model, device=device, 
                  epochs=config.fine_tune_epochs, lr=config.fine_tune_lr, weight_decay=config.fine_tune_weight_decay, gamma=config.fine_tune_gamma,
                  logging_enabled=config.logging_enabled, checkpointing_enabled=config.checkpointing_enabled)

    if config.should_evaluate:
        test_dataloader = prepare_dataset(config.test_path, config.test_batch_size, is_train=False, distributed=False)
        evaluate(dataloader=test_dataloader, model=model, device=device,
                 logging_enabled=config.logging_enabled)

    if config.should_visualize:
        visualization_dataloader = prepare_dataset(config.visualization_path, config.visualization_batch_size, is_train=False, distributed=False)
        visualize(dataloader=visualization_dataloader, model=model, device=device, 
                  num_samples=config.visualization_samples)

def distributed_experiment(rank, world_size, config, experiment_id):
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

        if config.should_pre_train:
            pre_train_dataloader = prepare_dataset(config.pre_train_path, config.pre_train_batch_size, is_train=True, distributed=True, rank=rank, world_size=world_size)
            pre_train(dataloader=pre_train_dataloader, model=model, device=rank, 
                      epochs=config.pre_train_epochs, lr=config.pre_train_lr, weight_decay=config.pre_train_weight_decay, gamma=config.pre_train_gamma,
                      logging_enabled=config.logging_enabled, checkpointing_enabled=config.checkpointing_enabled)
    
        if config.should_fine_tune:
            fine_tune_dataloader = prepare_dataset(config.fine_tune_path, config.fine_tune_batch_size, is_train=True, distributed=True, rank=rank, world_size=world_size)
            fine_tune(dataloader=fine_tune_dataloader, model=model, device=rank, 
                      epochs=config.fine_tune_epochs, lr=config.fine_tune_lr, weight_decay=config.fine_tune_weight_decay, gamma=config.fine_tune_gamma,
                      logging_enabled=config.logging_enabled, checkpointing_enabled=config.checkpointing_enabled)

        if config.should_evaluate:
            test_dataloader = prepare_dataset(config.test_path, config.test_batch_size, is_train=False, distributed=True, rank=rank, world_size=world_size)
            evaluate(dataloader=test_dataloader, model=model, device=rank,
                     logging_enabled=config.logging_enabled)

        if config.should_visualize:
            visualization_dataloader = prepare_dataset(config.visualization_path, config.visualization_batch_size, is_train=False, distributed=True, rank=rank, world_size=world_size)
            visualize(dataloader=visualization_dataloader, model=model, device=rank, 
                      num_samples=config.visualization_samples)
    
    finally:
        dist.barrier()
        dist.destroy_process_group()

def multi_device_experiment(config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = torch.cuda.device_count()
    experiment_id = str(uuid.uuid4())
    mp.spawn(distributed_experiment, args=(world_size, config, experiment_id,), nprocs=world_size, join=True)

if __name__ == '__main__':
    config = ExperimentConfig(
        data_parallel=True,
        logging_enabled=False,
        checkpointing_enabled=True,
        initialize_from_checkpoint=True,
        should_pre_train=False,
        should_fine_tune=False,
        should_evaluate=True,
        should_visualize=False,
        pre_train_path='../data1/waymo_dataset/v1.1/tensor_cache/training',
        fine_tune_path='../data1/waymo_dataset/v1.1/tensor_cache/training',
        test_path='../data1/waymo_dataset/v1.1/tensor_cache/validation',
        visualization_path='../data1/waymo_dataset/v1.1/tensor_cache/validation',
        pre_train_batch_size=16,
        pre_train_epochs=100,
        pre_train_lr=1e-4,
        pre_train_weight_decay=0,
        pre_train_gamma=0.999,
        fine_tune_batch_size=1,
        fine_tune_epochs=100,
        fine_tune_lr=1e-5,
        fine_tune_weight_decay=0,
        fine_tune_gamma=0.999,
        test_batch_size=1,
        visualization_batch_size=1,
        visualization_samples=100,
        road_map_image_size=256,
        road_map_window_size=8,
        trajectory_feature_dim=10,
        embedding_dim=256,
        flow_field_hidden_dim=256,
        flow_field_fourier_features=0#128
    )

    if config.logging_enabled:
        wandb.login()

    if torch.cuda.is_available() and config.data_parallel:
        multi_device_experiment(config)
    else:
        single_device_experiment(config)