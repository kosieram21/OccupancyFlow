import torch
import torch.distributed as dist
from datasets.Waymo import get_image_velocity

def end_point_error(target_flow, estimated_flow):
    l2_dist = torch.norm(estimated_flow - target_flow, p=2, dim=-1)
    epe = l2_dist.mean()
    return epe

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
            
            #flow_field_mask = flow_field_mask.view(-1)
            #flow = flow.view(-1, 2)[flow_field_mask == 1]
            #flow_field_velocities = flow_field_velocities.view(-1, 2)[flow_field_mask == 1]

            flow = flow[flow_field_mask]
            flow_field_velocities = flow_field_velocities[flow_field_mask]

            world_velocities = get_image_velocity(flow_field_velocities.cpu().numpy())
            world_velocities = torch.from_numpy(world_velocities).to(device)

            world_flow = get_image_velocity(flow.cpu().numpy())
            world_flow = torch.from_numpy(world_flow).to(device)

            batch_epe = end_point_error(world_velocities, world_flow)
            epe_sum += batch_epe
            count += 1

        epe_score = epe_sum / count
        epe_score = aggregate_epe(epe_score)

    return epe_score
