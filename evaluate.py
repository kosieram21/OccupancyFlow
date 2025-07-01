import torch
import torch.distributed as dist
from datasets.Waymo import get_image_velocity

def end_point_error(target_flow, estimated_flow, mask=None):
    scenes_in_batch = target_flow.shape[0]
    l2_distance = torch.norm(estimated_flow - target_flow, p=2, dim=-1)
    if mask is not None:
        l2_distance = l2_distance * mask
        sum_per_scene = l2_distance.sum(dim=-1)
        count_per_scene = mask.sum(dim=-1)
        total_agents += count_per_scene.sum()
        scene_epe = sum_per_scene / count_per_scene
        total_epe = scene_epe.sum()
    else:
        total_epe = l2_distance.sum()
    return total_epe, scenes_in_batch

# TODO: should this be moved to a shared location?
def aggregate_epe(epe):
    total_epe = epe.clone()

    if dist.is_initialized():
        dist.all_reduce(total_epe, op=dist.ReduceOp.SUM)
        total_epe /= dist.get_world_size()

    return total_epe.item()

def evaluate(dataloader, model, device):
    model.eval()

    with torch.no_grad():
        epe_sum = 0
        count = 0

        for batch in dataloader:
            road_map, agent_trajectories, \
            flow_field_agent_ids, flow_field_positions, flow_field_times, flow_field_velocities, \
            agent_mask, flow_field_mask = batch

            road_map = road_map.to(device)
            agent_trajectories = agent_trajectories.to(device)
            flow_field_positions = flow_field_positions.to(device)
            flow_field_times = flow_field_times.to(device)
            flow_field_velocities = flow_field_velocities.to(device)
            agent_mask = agent_mask.to(device)
            flow_field_mask = flow_field_mask.to(device)

            flow, _ = model(flow_field_times, flow_field_positions, road_map, agent_trajectories, agent_mask)
            flow_test, _ = model(flow_field_times[0][flow_field_mask[0]].unsqueeze(0),
                                 flow_field_positions[0][flow_field_mask[0]].unsqueeze(0),
                                 road_map[0].unsqueeze(0), agent_trajectories[0].unsqueeze(0),
                                 agent_mask[0].unsqueeze(0))

            world_velocities = get_image_velocity(flow_field_velocities.cpu().numpy())
            world_velocities = torch.from_numpy(world_velocities).to(device)

            world_flow = get_image_velocity(flow.cpu().numpy())
            world_flow = torch.from_numpy(world_flow).to(device)

            batch_epe, scenes_in_batch = end_point_error(world_velocities, world_flow, flow_field_mask)
            epe_sum += batch_epe
            count += scenes_in_batch

        epe_score = epe_sum / count
        epe_score = aggregate_epe(epe_score)

    return epe_score
