import torch
import torch.distributed as dist

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
            road_map, agent_trajectories, unobserved_positions, future_times, \
            target_velocity, agent_mask, flow_field_mask = batch

            road_map = road_map.to(device)
            agent_trajectories = agent_trajectories.to(device)
            unobserved_positions = unobserved_positions.to(device)
            future_times = future_times.to(device)
            target_velocity = target_velocity.to(device)
            agent_mask = agent_mask.to(device)
            flow_field_mask = flow_field_mask.to(device)

            flow = model(future_times, unobserved_positions, road_map, agent_trajectories, agent_mask)
            
            flow_field_mask = flow_field_mask.view(-1)
            flow = flow.view(-1, 2)[flow_field_mask == 1]
            target_velocity = target_velocity.view(-1, 2)[flow_field_mask == 1]

            batch_epe = end_point_error(target_velocity, flow)
            epe_sum += batch_epe
            count += 1

        epe_score = epe_sum / count
        epe_score = aggregate_epe(epe_score)

    return epe_score
