import os
import torch
import torch.nn.functional as F

def train(dataloader, model, epochs, lr, weight_decay, gamma, device):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    # TODO: something is wrong with the data. we need to be able to just fit a flow field for a single data sample...
    road_map, agent_trajectories, unobserved_positions, future_times, target_velocity, target_occupancy_grid = next(iter(dataloader))
    for epoch in range(1000):#range(epochs):
        epoch_loss = 0
        num_batches = 0
        #for road_map, agent_trajectories, unobserved_positions, future_times, target_velocity, target_occupancy_grid in dataloader:
        road_map = road_map.to(device)
        agent_trajectories = agent_trajectories.to(device)
        unobserved_positions = unobserved_positions.to(device)
        future_times = future_times.to(device)
        target_velocity = target_velocity.to(device)
        target_occupancy_grid = target_occupancy_grid.to(device)

        flow = model(future_times, unobserved_positions, road_map, agent_trajectories)
        loss = F.mse_loss(flow, target_velocity)
        print(f'Batch {num_batches+1}, Loss: {loss}')

        optim.zero_grad()
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
        num_batches += 1
        # end for

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')