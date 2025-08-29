import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.WaymoScene import WaymoScene

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
            sample['flow_field_positions'],
            sample['flow_field_times'],
            sample['flow_field_velocities'],
            sample['flow_field_agnet_ids'],
            sample['occupancy_grid_positions'],
            sample['occupancy_grid_times'],
            sample['occupancy_grid_unoccluded_occupancies'],
            sample['occupancy_grid_occluded_occupancies'],
            sample['agent_mask'],
            sample['flow_field_mask']
        )

def cache_scene(path,
                road_map, agent_trajectories, 
                flow_field_positions, flow_field_times, flow_field_velocities, flow_field_agent_ids,
                occupancy_grid_positions, occupancy_grid_times, occupancy_grid_unoccluded_occupancies, occupancy_grid_occluded_occupancies, 
                agent_mask, flow_field_mask):
    scene = {
        'road_map': road_map,
        'agent_trajectories': agent_trajectories,
        'flow_field_positions': flow_field_positions,
        'flow_field_times': flow_field_times,
        'flow_field_velocities': flow_field_velocities,
        'flow_field_agnet_ids': flow_field_agent_ids,
        'occupancy_grid_positions': occupancy_grid_positions,
        'occupancy_grid_times': occupancy_grid_times,
        'occupancy_grid_unoccluded_occupancies': occupancy_grid_unoccluded_occupancies,
        'occupancy_grid_occluded_occupancies': occupancy_grid_occluded_occupancies,
        'agent_mask': agent_mask,
        'flow_field_mask': flow_field_mask,
    }
    torch.save(scene, path)

def cache_data(dataloader, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    idx = 0
    for scene in dataloader:
        road_map = scene.observed_state.road_map
        agent_trajectories = scene.observed_state.agent_trajectories

        flow_field_positions = scene.flow_field.positions
        flow_field_times = scene.flow_field.times
        flow_field_velocities = scene.flow_field.velocities
        flow_field_agent_ids = scene.flow_field.agent_ids

        occupancy_grid_positions = scene.occupancy_grid.positions
        occupancy_grid_times = scene.occupancy_grid.times
        occupancy_grid_unoccluded_occupancies = scene.occupancy_grid.unoccluded_occupancies
        occupancy_grid_occluded_occupancies = scene.occupancy_grid.occluded_occupancies

        agent_mask = scene.observed_state.agent_mask
        flow_field_mask = scene.flow_field.flow_mask

        for i in range(road_map.shape[0]):
            idx += 1
            path = os.path.join(cache_dir, f'sample{idx:06d}.pt')
            cache_scene(path,
                        road_map[i], agent_trajectories[i],
                        flow_field_positions[i], flow_field_times[i], flow_field_velocities[i], flow_field_agent_ids[i],
                        occupancy_grid_positions[i], occupancy_grid_times[i], occupancy_grid_unoccluded_occupancies[i], occupancy_grid_occluded_occupancies[i],
                        agent_mask[i], flow_field_mask[i])
            print(idx) # TODO: delete me
            
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

    flow_field_agent_ids = []
    flow_field_positions = []
    flow_field_times = []
    flow_field_velocities = []

    occupancy_grid_positions = []
    occupancy_grid_times = []
    occupancy_grid_unoccluded_occupancies = []
    occupancy_grid_occluded_occupancies = []

    for scene in batch:
        road_map, trajectories, \
        positions, times, velocities, ids, \
        grid_points, grid_times, occupancy_grid, occluded_occupancy_grid,\
        _, _ = scene

        road_maps.append(road_map)
        agent_trajectories.append(trajectories)

        flow_field_agent_ids.append(ids)
        flow_field_positions.append(positions)
        flow_field_times.append(times)
        flow_field_velocities.append(velocities)

        occupancy_grid_positions.append(grid_points)
        occupancy_grid_times.append(grid_times)
        occupancy_grid_unoccluded_occupancies.append(occupancy_grid)
        occupancy_grid_occluded_occupancies.append(occluded_occupancy_grid)

    max_agents = max(t.shape[0] for t in agent_trajectories)
    max_agent_positions = max(p.shape[0] for p in flow_field_positions)
    agent_trajectories, agent_mask = pad_tensors(agent_trajectories, max_agents)
    flow_field_agent_ids, flow_field_mask = pad_tensors(flow_field_agent_ids, max_agent_positions)
    flow_field_positions, _ = pad_tensors(flow_field_positions, max_agent_positions)
    flow_field_times, _ = pad_tensors(flow_field_times, max_agent_positions)
    flow_field_velocities, _ = pad_tensors(flow_field_velocities, max_agent_positions)

    road_map_batch = torch.stack(road_maps, dim=0)
    agent_trajectories_batch = torch.stack(agent_trajectories, dim=0)

    flow_field_positions_batch = torch.stack(flow_field_positions, dim=0)
    flow_field_times_batch = torch.stack(flow_field_times, dim=0)
    flow_field_velocities_batch = torch.stack(flow_field_velocities, dim=0)
    flow_field_agent_ids_batch = torch.stack(flow_field_agent_ids, dim=0)
    
    occupancy_grid_positions_batch = torch.stack(occupancy_grid_positions, dim=0)
    occupancy_grid_times_batch = torch.stack(occupancy_grid_times, dim=0)
    occupancy_grid_unoccluded_occupancies_batch = torch.stack(occupancy_grid_unoccluded_occupancies, dim=0)
    occupancy_grid_occluded_occupancies_batch = torch.stack(occupancy_grid_occluded_occupancies, dim=0)

    agent_mask_batch = torch.stack(agent_mask, dim=0)
    flow_field_mask_batch = torch.stack(flow_field_mask, dim=0)

    # TODO: ODE fintuning data should be pre-collated as well
    waymo_scene = WaymoScene.from_tensors(
        road_map_batch, agent_trajectories_batch,
        flow_field_positions_batch, flow_field_times_batch, flow_field_velocities_batch, flow_field_agent_ids_batch,
        occupancy_grid_positions_batch, occupancy_grid_times_batch, occupancy_grid_unoccluded_occupancies_batch, occupancy_grid_occluded_occupancies_batch,
        agent_mask_batch, flow_field_mask_batch
    )

    return waymo_scene
            