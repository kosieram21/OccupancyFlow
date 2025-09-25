import wandb
import torch
import torch.distributed as dist
from datasets.Waymo import get_image_velocity

def end_point_error(target_flow, estimated_flow, mask=None):
    scenes_in_batch = target_flow.shape[0]
    l2_distance = torch.norm(estimated_flow - target_flow, p=2, dim=-1)
    if mask is not None:
        l2_distance = l2_distance * mask
        total_l2_distance = l2_distance.sum(dim=-1)
        agents_per_scene = mask.sum(dim=-1)
    else:
        total_l2_distance = l2_distance.sum(dim=-1)
        agents_per_scene = total_l2_distance.new_full((scenes_in_batch,), l2_distance.size(1))
    scene_epe = total_l2_distance / agents_per_scene
    total_epe = scene_epe.sum()
    return total_epe, scenes_in_batch

# TODO: should this be moved to a shared location?
def aggregate_epe(epe):
    total_epe = epe.clone()

    if dist.is_initialized():
        dist.all_reduce(total_epe, op=dist.ReduceOp.SUM)
        total_epe /= dist.get_world_size()

    return total_epe.item()

def evaluate(dataloader, model, device,
             logging_enabled=False):
    model.eval()

    if logging_enabled:
        wandb.define_metric("evaluate batch epe", step_metric="evaluate batch")
        wandb.define_metric("evaluate batch", hidden=True)

    with torch.no_grad():
        epe_sum = 0
        num_batches = 0
        num_scenes = 0

        for scene in dataloader:
            road_map = scene.observed_state.road_map
            agent_trajectories = scene.observed_state.agent_trajectories

            flow_field_agent_ids = scene.flow_field.agent_ids
            flow_field_positions = scene.flow_field.positions
            flow_field_times = scene.flow_field.times
            flow_field_velocities = scene.flow_field.velocities

            # TODO: use these in evaluation
            occupancy_grid_positions = scene.occupancy_grid.positions
            occupancy_grid_times = scene.occupancy_grid.times
            occupancy_grid_unoccluded_occupancies = scene.occupancy_grid.unoccluded_occupancies
            occupancy_grid_occluded_occupancies = scene.occupancy_grid.occluded_occupancies

            agent_mask = scene.observed_state.agent_mask
            flow_field_mask = scene.flow_field.flow_mask

            road_map = road_map.to(device)
            agent_trajectories = agent_trajectories.to(device)
            flow_field_positions = flow_field_positions.to(device)
            flow_field_times = flow_field_times.to(device)
            flow_field_velocities = flow_field_velocities.to(device)
            agent_mask = agent_mask.to(device)
            flow_field_mask = flow_field_mask.to(device)

            flow, _ = model(flow_field_times, flow_field_positions, road_map, agent_trajectories, agent_mask)

            world_velocities = get_image_velocity(flow_field_velocities.cpu().numpy())
            world_velocities = torch.from_numpy(world_velocities).to(device)

            world_flow = get_image_velocity(flow.cpu().numpy())
            world_flow = torch.from_numpy(world_flow).to(device)

            batch_epe, scenes_in_batch = end_point_error(world_velocities, world_flow, flow_field_mask)

            if logging_enabled:
                avg_batch_epe = aggregate_epe(batch_epe / scenes_in_batch)
                wandb.log({"eveluate batch epe": avg_batch_epe, "evaluate batch": num_batches})
                print(f'Batch {num_batches+1} (pre-train), EPE: {avg_batch_epe:.6f}')

            epe_sum += batch_epe
            num_batches += 1
            num_scenes += scenes_in_batch

        epe_score = epe_sum / num_scenes
        epe_score = aggregate_epe(epe_score)

    if logging_enabled:
        wandb.log({'epe': epe_score})
        print(f'end point error: {epe_score}')

    return epe_score
