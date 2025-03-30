import time # delete me
import torch
import torch.nn.functional as F

def train(dataloader, model, epochs, lr, weight_decay, gamma, device):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        total_time = 0 # delete me
        for road_map, agent_trajectories, unobserved_positions, future_times, target_velocity, agent_mask, flow_field_mask in dataloader:
            start = time.time() # delete me
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

            loss = F.mse_loss(flow, target_velocity)
            end = time.time() # delete me
            elapsed = end - start # delete me
            total_time += elapsed
            print(f'Batch {num_batches+1}, Loss: {loss}, Time: {elapsed}, Total Time: {total_time}')

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')