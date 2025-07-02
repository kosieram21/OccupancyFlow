import os
import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
from collections import defaultdict

def move_batch_to_device(batch, device):
    road_map, agent_trajectories, \
    flow_field_agent_ids, flow_field_positions, flow_field_times, flow_field_velocities, \
    agent_mask, flow_field_mask = batch
    
    road_map = road_map.to(device)
    agent_trajectories = agent_trajectories.to(device)
    flow_field_agent_ids = flow_field_agent_ids.to(device)
    flow_field_positions = flow_field_positions.to(device)
    flow_field_times = flow_field_times.to(device)
    flow_field_velocities = flow_field_velocities.to(device)
    agent_mask = agent_mask.to(device)
    flow_field_mask = flow_field_mask.to(device)

    return \
        road_map, agent_trajectories, \
        flow_field_agent_ids, flow_field_positions, flow_field_times, flow_field_velocities, \
        agent_mask, flow_field_mask

# TODO: should this be moved to a shared location?
def aggregate_loss(loss):
    total_loss = loss.clone()

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss /= dist.get_world_size()

    return total_loss.item()

def save_checkpoint(model, checkpoint_root, checkpoint_id):
    os.makedirs(checkpoint_root, exist_ok=True)
    params = model.module if dist.is_initialized() else model
    torch.save(params.state_dict(), os.path.join(checkpoint_root, f'{checkpoint_id}.pt'))

# TODO: I wonder if this should be part of the waymo collate function
def construct_agent_trajectories(agent_ids, positions, times, forecast_horizon):
    rounded_times = torch.round(times * 10) / 10.0
    unique_times = torch.unique(rounded_times)
    integration_times = unique_times[unique_times <= 9.0]

    trajectories = [{} for _ in range(forecast_horizon)]
    present = [defaultdict(lambda: False) for _ in range(forecast_horizon)]
    initial_values = [[] for _ in range(forecast_horizon)]

    agent_seen = set()
    agent_offsets = {}
    offset = 0

    for time_val in integration_times:
        mask = rounded_times == time_val
        time_indices = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
        agent_ids_at_time = agent_ids[time_indices]
        time_index = int((time_val * 10).item())

        unique_ids = torch.unique(agent_ids_at_time)
        for agent_id in unique_ids.tolist():
            agent_mask = agent_ids_at_time == agent_id
            agent_indices = torch.nonzero(agent_mask.flatten(), as_tuple=False).squeeze(1)
            global_indices = time_indices[agent_indices]
            agent_positions = positions[global_indices]

            trajectories[time_index][agent_id] = agent_positions
            present[time_index][agent_id] = True

            if agent_id not in agent_seen:
                agent_seen.add(agent_id)
                initial_values[time_index].append(agent_positions)
                num_agent_positions = agent_positions.shape[0]
                start = offset
                end = offset + num_agent_positions
                agent_offsets[agent_id] = (start, end)
                offset = end

    return trajectories, present, initial_values, agent_offsets, integration_times, list(agent_seen)

def reconstruct_trajectories(estimated_occupancy, present, agent_offsets, integration_times, agent_ids, forecast_horizon):
    reconstructed_trajectories = [{} for _ in range(forecast_horizon)]
    for time_index, _ in enumerate(integration_times):
        for id in agent_ids:
            if present[time_index][id]:
                start, end = agent_offsets[id]
                estimated_occupancy_at_time = estimated_occupancy[time_index][0]
                reconstructed_trajectories[time_index][id] = estimated_occupancy_at_time[start:end]
    return reconstructed_trajectories

def flow_matching(model, times, positions, velocities, road_map, agent_trajectories, agent_mask=None, flow_field_mask=None):
    scenes_in_batch = velocities.shape[0]
    flow, scene_context = model(times, positions, road_map, agent_trajectories, agent_mask)
    se = (flow - velocities)**2 
    se = se.sum(dim=-1)
    if flow_field_mask is not None:
        se = se * flow_field_mask
        total_se = se.sum(dim=-1)
        agents_per_scene = flow_field_mask.sum(dim=-1)
    else:
        total_se = se.sum(dim=-1)
        agents_per_scene = total_se.new_full((scenes_in_batch,), se.size(1))
    scene_mse = total_se / agents_per_scene
    total_mse = scene_mse.sum()
    mse = total_mse / scenes_in_batch
    return mse, scene_context

def occupancy_alignment(flow_field, agent_ids, positions, times, scene_context, flow_field_mask, forecast_horizon=91):
    loss = 0
    count = 0

    num_scenes = agent_ids.shape[0]
    for scene_index in range(num_scenes):
        scene_mask = flow_field_mask[scene_index]
        ids = agent_ids[scene_index][scene_mask]
        p = positions[scene_index][scene_mask]
        t = times[scene_index][scene_mask]
        context = scene_context[scene_index].unsqueeze(0)

        loss += scene_occupancy_alignment(flow_field, ids, p, t, context, forecast_horizon)
        count += 1
    
    return loss / count
    

def scene_occupancy_alignment(flow_field, agent_ids, positions, times, scene_context, forecast_horizon=91):
    trajectories, present, initial_values, agent_offsets, integration_times, ids = construct_agent_trajectories(agent_ids, positions, times, forecast_horizon)
    estimated_occupancy = flow_field.solve_ivp(initial_values, integration_times, scene_context)
    estimated_trajectories = reconstruct_trajectories(estimated_occupancy, present, agent_offsets, integration_times, ids, forecast_horizon)
    
    loss = 0
    count = 0

    for time_index, _ in enumerate(integration_times):
        for id in ids:
            if present[time_index][id]:
                ground_truth_positions = trajectories[time_index][id]
                estimated_positions = estimated_trajectories[time_index][id]
                if ground_truth_positions.shape == estimated_positions.shape:
                    loss += torch.mean(torch.abs(ground_truth_positions - estimated_positions))
                    count += 1
                
    return loss / count

def pre_train(dataloader, model, device,
              epochs, lr, weight_decay, gamma, 
              logging_enabled=False, checkpointing_enabled=False):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    if logging_enabled:
        wandb.define_metric("pre-train batch loss", step_metric="pre-train batch")
        wandb.define_metric("pre-train epoch loss", step_metric="pre-train epoch")
        wandb.define_metric("pre-train batch", hidden=True)
        wandb.define_metric("pre-train epoch", hidden=True)

    total_batches = 0
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0.0, device=device)
        num_batches = 0

        for batch in dataloader:
            road_map, agent_trajectories, \
            flow_field_agent_ids, flow_field_positions, flow_field_times, flow_field_velocities, \
            agent_mask, flow_field_mask = move_batch_to_device(batch, device)

            loss, _ = flow_matching(model, 
                                    flow_field_times, flow_field_positions, flow_field_velocities, 
                                    road_map, agent_trajectories, 
                                    agent_mask, flow_field_mask)
            
            total_loss = aggregate_loss(loss.detach())

            if logging_enabled:
                wandb.log({"pre-train batch loss": total_loss, "pre-train batch": total_batches})
                print(f'Batch {total_batches+1} (pre-train), Loss: {total_loss:.6f}')

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            epoch_loss += total_loss
            num_batches += 1
            total_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        
        if logging_enabled:
            wandb.log({"pre-train epoch loss": avg_loss, "pre-train epoch": epoch})
            print(f'Epoch {epoch+1}/{epochs} (pre-train), Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        if checkpointing_enabled:
            save_checkpoint(model, f'checkpoints/pretrain', f'occupancy_flow_checkpoint{epoch}')

def fine_tune(dataloader, model, device,
              epochs, lr, weight_decay, gamma,
              logging_enabled=False, checkpointing_enabled=False):
    model.train()

    scene_encoder = model.module.scene_encoder if dist.is_initialized() else model.scene_encoder
    flow_field = model.module.flow_field if dist.is_initialized() else model.flow_field

    for param in scene_encoder.parameters():
        param.requires_grad = False

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    if logging_enabled:
        wandb.define_metric("fine-tune batch loss", step_metric="fine-tune batch")
        wandb.define_metric("fine-tune batch flow loss", step_metric="fine-tune batch")
        wandb.define_metric("fine-tune batch occupancy loss", step_metric="fine-tune batch")
        wandb.define_metric("fine-tune epoch loss", step_metric="fine-tune epoch")
        wandb.define_metric("fine-tune epoch flow loss", step_metric="fine-tune epoch")
        wandb.define_metric("fine-tune epoch occupancy loss", step_metric="fine-tune epoch")
        wandb.define_metric("fine-tune batch", hidden=True)
        wandb.define_metric("fine-tune epoch", hidden=True)

    total_batches = 0
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0.0, device=device)
        epoch_flow_loss = torch.tensor(0.0, device=device)
        epoch_occupancy_loss = torch.tensor(0.0, device=device)
        num_batches = 0

        for batch in dataloader:
            road_map, agent_trajectories, \
            flow_field_agent_ids, flow_field_positions, flow_field_times, flow_field_velocities, \
            agent_mask, flow_field_mask = move_batch_to_device(batch, device)

            with torch.no_grad():
                scene_context = scene_encoder(road_map, agent_trajectories, agent_mask)
            
            flow_loss, scene_context = flow_matching(model, 
                                                     flow_field_times, flow_field_positions, flow_field_velocities,
                                                     road_map, agent_trajectories, 
                                                     agent_mask, flow_field_mask)

            occupancy_loss = occupancy_alignment(flow_field, 
                                                 flow_field_agent_ids, flow_field_positions, flow_field_times,
                                                 scene_context,
                                                 flow_field_mask,
                                                 forecast_horizon=91)
            
            loss = flow_loss + occupancy_loss

            #TODO: can we do this with only one call to nccl?
            total_loss = aggregate_loss(loss.detach())
            total_flow_loss = aggregate_loss(flow_loss.detach())
            total_occupancy_loss = aggregate_loss(occupancy_loss.detach())

            if logging_enabled:
                wandb.log({
                    "fine-tune batch loss": total_loss,
                    "fine-tune batch flow loss": total_flow_loss,
                    "fine-tune batch occupancy loss": total_occupancy_loss, 
                    "fine-tune batch": total_batches
                })
                print(f'Batch {total_batches+1} (fine-tune), Loss: {total_loss:.6f}, Flow Loss: {total_flow_loss:.6f}, Occupancy Loss: {total_occupancy_loss:.6f}')

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            epoch_loss += total_loss
            epoch_flow_loss += total_flow_loss
            epoch_occupancy_loss += total_occupancy_loss
            num_batches += 1
            total_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_flow_loss = epoch_flow_loss / num_batches
        avg_occupancy_loss = epoch_occupancy_loss / num_batches
        
        if logging_enabled:
            wandb.log({
                "fine-tune epoch loss": avg_loss, 
                "fine-tune epoch flow loss": avg_flow_loss,
                "fine-tune epoch occupancy loss": avg_occupancy_loss,
                "fine-tune epoch": epoch
            })
            print(f'Epoch {epoch+1}/{epochs} (fine-tune), Loss: {avg_loss:.6f}, Flow Loss: {avg_flow_loss:.6f}, Occupancy Loss: {avg_occupancy_loss:.6f} LR: {scheduler.get_last_lr()[0]:.6f}')

        if checkpointing_enabled:
            save_checkpoint(model, f'checkpoints/finetune', f'occupancy_flow_checkpoint{epoch}')