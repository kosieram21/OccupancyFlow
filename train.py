import os
import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist

# TODO: should this be moved to a shared location?
def aggregate_loss(loss):
    total_loss = loss.clone()

    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_loss /= dist.get_world_size()

    return total_loss.item()

def train(dataloader, model, device,
          epochs, lr, weight_decay, gamma, 
          logging_enabled=False, checkpointing_enabled=False):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    if logging_enabled:
        wandb.define_metric("batch loss", step_metric="batch")
        wandb.define_metric("epoch loss", step_metric="epoch")
        wandb.define_metric("batch", hidden=True)
        wandb.define_metric("epoch", hidden=True)

    total_batches = 0
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0.0, device=device)
        num_batches = 0

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

            flow = model(flow_field_times, flow_field_positions, road_map, agent_trajectories, agent_mask)
            
            flow_field_mask = flow_field_mask.view(-1)
            flow = flow.view(-1, 2)[flow_field_mask == 1]
            flow_field_velocities = flow_field_velocities.view(-1, 2)[flow_field_mask == 1]

            loss = F.mse_loss(flow, flow_field_velocities)
            total_loss = aggregate_loss(loss.detach())

            if logging_enabled:
                wandb.log({"batch loss": total_loss, "batch": total_batches})
                print(f'Batch {total_batches+1}, Loss: {total_loss:.6f}')

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            epoch_loss += loss.detach()
            num_batches += 1
            total_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        total_avg_loss = aggregate_loss(avg_loss)
        
        if logging_enabled:
            wandb.log({"epoch loss": total_avg_loss, "epoch": epoch})
            print(f'Epoch: {epoch+1}/{epochs}, Loss: {total_avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        if checkpointing_enabled:
            checkpoint_root = 'checkpoints'
            os.makedirs(checkpoint_root, exist_ok=True)
            params = model.module if dist.is_initialized() else model
            torch.save(params.state_dict(), os.path.join(checkpoint_root, f'occupancy_flow_checkpoint{epoch}.pt'))