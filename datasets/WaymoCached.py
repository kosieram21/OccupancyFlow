import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class WaymoCached(Dataset):
    def __init__(self, cache_dir):
        self.file_paths = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(cache_dir)
            for f in files if f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sample = torch.load(file_path, map_location='cpu')
        return (
            sample['road_map'],
            sample['agent_trajectories'],
            sample['unobserved_positions'],
            sample['future_times'],
            sample['target_velocity'],
            sample['agent_mask'],
            sample['flow_field_mask']
        )

def cache_scene(road_map, agent_trajectories, 
                unobserved_positions, future_times, target_velocity, 
                agent_mask, flow_field_mask, path):
    scene = {
        'road_map': road_map,
        'agent_trajectories': agent_trajectories,
        'unobserved_positions': unobserved_positions,
        'future_times': future_times,
        'target_velocity': target_velocity,
        'agent_mask': agent_mask,
        'flow_field_mask': flow_field_mask,
    }
    torch.save(scene, path)

def cache_data(dataloader, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    idx = 0
    for road_map, agent_trajectories, unobserved_positions, future_times, target_velocity, agent_mask, flow_field_mask in dataloader:
        for i in range(road_map.shape[0]):
            idx += 1
            path = os.path.join(cache_dir, f'sample{idx:06d}.pt')
            cache_scene(road_map[i], agent_trajectories[i],
                        unobserved_positions[i], future_times[i], target_velocity[i],
                        agent_mask[i], flow_field_mask[i], path)
            print(idx)
            
# TODO: move this to a shared location
def pad_tensors(tensors, max_size):
    padded_tensors = []
    masks = []

    for tensor in tensors:
        samples = tensor.shape[0]
        padding_size = max_size - samples
        
        padding = [0] * (2 * tensor.dim())
        padding[-1] = padding_size
        padded_tensor = F.pad(tensor, padding)
        padded_tensors.append(padded_tensor)
        
        mask = torch.ones(max_size, dtype=torch.bool)
        mask[samples:] = 0
        masks.append(mask)

    return padded_tensors, masks
            
def waymo_cached_collate_fn(batch):
    road_maps = []
    agent_trajectories = []
    unobserved_positions = []
    future_times = []
    future_velocities = []

    for road_map, trajectories, positions, times, velocities, _, _ in batch:
        road_maps.append(road_map)
        agent_trajectories.append(trajectories)
        unobserved_positions.append(positions)
        future_times.append(times)
        future_velocities.append(velocities)

    max_agents = max(t.shape[0] for t in agent_trajectories)
    max_unobserved_positions = max(t.shape[0] for t in unobserved_positions)
    agent_trajectories, agent_mask = pad_tensors(agent_trajectories, max_agents)
    unobserved_positions, flow_field_mask = pad_tensors(unobserved_positions, max_unobserved_positions)
    future_times, _ = pad_tensors(future_times, max_unobserved_positions)
    future_velocities, _ = pad_tensors(future_velocities, max_unobserved_positions)

    road_map_batch = torch.stack(road_maps, dim=0)
    agent_trajectories_batch = torch.stack(agent_trajectories, dim=0)
    unobserved_positions_batch = torch.stack(unobserved_positions, dim=0)
    future_times_batch = torch.stack(future_times, dim=0)
    future_velocities_batch = torch.stack(future_velocities, dim=0)
    agent_mask_batch = torch.stack(agent_mask, dim=0)
    flow_field_mask_batch = torch.stack(flow_field_mask, dim=0)

    return (
        road_map_batch,
        agent_trajectories_batch,
        unobserved_positions_batch,
        future_times_batch,
        future_velocities_batch,
        agent_mask_batch,
        flow_field_mask_batch
    )
            