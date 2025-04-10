import os
import torch

def cache_data(dataloader, cache_dir):
    idx = 0
    for road_map, agent_trajectories, unobserved_positions, future_times, target_velocity, agent_mask, flow_field_mask in enumerate(dataloader):
        for i in range(road_map.shape[0]):
            idx += 1
            scene = {
                'road_map': road_map[i],
                'agent_trajectories': agent_trajectories[i],
                'unobserved_positions': unobserved_positions[i],
                'future_times': future_times[i],
                'target_velocity': target_velocity[i],
                'agent_mask': agent_mask[i],
                'flow_field_mask': flow_field_mask[i],
            }
            torch.save(scene, os.path.join(cache_dir, f'sample{idx:06d}.pt'))