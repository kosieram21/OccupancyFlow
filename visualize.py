import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from datasets.Waymo import get_image_coordinates, get_world_coordinates, get_image_velocity, rotate_points_around_origin

def render_observed_scene_state(road_map, agent_trajectories, save_path=None):
    image_buffer = road_map.numpy() / 255.0
    
    fig, ax = plt.subplots()
    ax.imshow(image_buffer)
    ax.set_xlim(0, image_buffer.shape[1])
    ax.set_ylim(image_buffer.shape[0], 0)
    ax.invert_yaxis()
    ax.axis('off')

    agent_cmap = ['blue', 'orange', 'yellow', 'purple']
    track_size_map = [5, 1, 1, 1]

    for agent in range(agent_trajectories.shape[0]):
        agent_trajectory = get_image_coordinates(agent_trajectories[agent,:,:2])
        agent_x = agent_trajectories[agent, -1, 0].item()
        agent_y = agent_trajectories[agent, -1, 1].item()
        agent_bbox_yaw = agent_trajectories[agent, -1, 2].item()
        agent_width = agent_trajectories[agent, -1, 3].item()
        agent_length = agent_trajectories[agent, -1, 4].item()
        agent_type = agent_trajectories[agent,-1, -1].item()
        agent_type = 4 if math.isnan(agent_type) else int(agent_type)
        agent_color = agent_cmap[agent_type - 1]
        agent_track_size = track_size_map[agent_type - 1]
        
        segments = np.concatenate([agent_trajectory[:-1, None], agent_trajectory[1:, None]], axis=1)
        alphas = np.linspace(0.2, 1.0, len(segments))
        colors = np.array([[0, 1, 1, a] for a in alphas])
        lc = LineCollection(segments, colors=colors, linewidths=agent_track_size)
        ax.add_collection(lc)
    
        half_w = agent_width / 2
        half_l = agent_length / 2
        corners = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
        corners = rotate_points_around_origin(corners, agent_bbox_yaw)
        corners += np.array([agent_x, agent_y])
        corners = get_image_coordinates(corners)
        ax.plot(
            [*corners[:, 0], corners[0, 0]],
            [*corners[:, 1], corners[0, 1]],
            color=agent_color, linewidth=1
        )

        heading = np.array([[0.75 * agent_length, 0]])
        heading = rotate_points_around_origin(heading, agent_bbox_yaw)
        heading = get_image_velocity(heading)[0]
        center = get_image_coordinates(np.array([[agent_x, agent_y]]))[0]
        ax.arrow(center[0], center[1],
                 heading[0], heading[1],
                 head_width=1.0, head_length=1.5,
                 fc='orange', ec='orange')
        
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close(fig)

def render_flow_at_spacetime(road_map, times, positions, velocity, save_path=None):
    image_buffer = road_map.numpy() / 255.0

    groups = defaultdict(list)
    [groups[round(val.item(), 1)].append(idx) for idx, val in enumerate(times)]
    sorted_keys = sorted(groups.keys())

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.imshow(image_buffer)
        ax.set_xlim(0, image_buffer.shape[1])
        ax.set_ylim(image_buffer.shape[0], 0)
        ax.invert_yaxis()
        ax.axis('off')

        time = list(sorted_keys)[frame]
        indices = groups[time]
    
        group_positions = get_image_coordinates(positions[indices])
        group_velocity = get_image_velocity(velocity[indices])

        ax.set_title(f"Future Flow (t = {time}s)")
    
        x_coords, y_coords = zip(*group_positions)
        ax.scatter(x_coords, y_coords, marker='o', s=5, color='blue')
        ax.quiver(x_coords, y_coords, group_velocity[:, 0], group_velocity[:, 1], 
                  angles='xy', scale_units='xy', scale=4.0, 
                  color='orange', width=0.007, 
                  headwidth=4, headlength=5, headaxislength=3)

    anim = FuncAnimation(fig, update, frames=len(groups), repeat=False)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, fps=10, dpi=300)

    plt.close(fig)

    return anim

def render_flow_at_ground_truth_occupancy(model, road_map,
                                          times, positions,
                                          scene_context,
                                          save_path=None):
    estimated_flow_at_ground_truth_occupancy = model.flow_field(times, positions, scene_context)
    return render_flow_at_spacetime(road_map[0].cpu(), 
                                    times[0].cpu(), 
                                    positions[0].cpu(), 
                                    estimated_flow_at_ground_truth_occupancy[0].detach().cpu(), 
                                    save_path=save_path)
    
def render_occupancy_and_flow_unoccluded(model, road_map,
                                         times, positions, 
                                         initial_time, 
                                         scene_context,
                                         save_path=None):
    groups = defaultdict(list)
    [groups[round(val.item(), 1)].append(idx) for idx, val in enumerate(times[0])]
    sorted_keys = sorted(groups.keys())
    indices = groups[sorted_keys[initial_time]]

    initial_occupancy = positions[0][indices].unsqueeze(0)
    integration_times = torch.FloatTensor(sorted_keys[initial_time:]).to(road_map.device)
    estimated_occupancy = model.warp_occupancy(initial_occupancy, integration_times, scene_context)

    factored_positions = []
    factored_times = []
    for i in range(len(estimated_occupancy)):
        p = estimated_occupancy[i]
        t = integration_times[i].view(1, 1, 1).expand(1, p.shape[1], 1)
        factored_positions.append(p[0])
        factored_times.append(t[0])

    factored_positions = torch.stack(factored_positions).view(1, -1, 2)
    factored_times = torch.stack(factored_times).view(1, -1, 1)

    estimated_flow_at_initial_occupancy = model.flow_field(factored_times, factored_positions, scene_context)
    return render_flow_at_spacetime(road_map[0].cpu(), 
                                    factored_times[0].detach().cpu(), 
                                    factored_positions[0].detach().cpu(), 
                                    estimated_flow_at_initial_occupancy[0].detach().cpu(), 
                                    save_path=save_path)
    
def render_flow_field(model, road_map, 
                      grid_size, stride, timesteps, freq, 
                      scene_context,
                      save_path=None):
    y_coords = np.arange(0, grid_size, stride)
    x_coords = np.arange(0, grid_size, stride)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    grid_points = get_world_coordinates(grid_points)
    grid_points = torch.FloatTensor(grid_points)

    num_cells = grid_points.shape[0]

    grid_points = grid_points.repeat(timesteps, 1)
    grid_points = grid_points.reshape(-1, 2).unsqueeze(0)

    grid_times = [t / freq for t in range(timesteps)]
    grid_times = torch.FloatTensor(grid_times)
    grid_times = grid_times.repeat_interleave(num_cells).unsqueeze(0).unsqueeze(-1)

    grid_points = grid_points.to(road_map.device)
    grid_times = grid_times.to(road_map.device)

    estimated_flow_at_grid = model.flow_field(grid_times, grid_points, scene_context)
    return render_flow_at_spacetime(road_map[0].cpu(), 
                                    grid_times[0].detach().cpu(), 
                                    grid_points[0].detach().cpu(), 
                                    estimated_flow_at_grid[0].detach().cpu(), 
                                    save_path=save_path)

def visualize(dataloader, model, device, 
              num_samples):
    samples_processed = 0

    for batch in dataloader:
        road_map, agent_trajectories, \
        flow_field_agent_ids, flow_field_positions, flow_field_times, flow_field_velocities, \
        agent_mask, flow_field_mask = batch

        for i in range(road_map.shape[0]):
            samples_processed += 1
            if samples_processed > num_samples:
                return
        
            sample_road_map = road_map[i].unsqueeze(0).to(device)
            sample_agent_trajectories = agent_trajectories[i].unsqueeze(0).to(device)
            sample_flow_field_positions = flow_field_positions[i].unsqueeze(0).to(device)
            sample_flow_field_times = flow_field_times[i].unsqueeze(0).to(device)
            sample_flow_field_velocities = flow_field_velocities[i].unsqueeze(0).to(device)

            scene_context = model.scene_encoder(sample_road_map, sample_agent_trajectories)

            root = f'visualization/sample{samples_processed}'

            render_observed_scene_state(road_map=sample_road_map[0].cpu(), 
                                        agent_trajectories=sample_agent_trajectories[0].cpu(), 
                                        save_path=f'{root}/observed_scene_state.png')
        
            render_flow_at_spacetime(road_map=sample_road_map[0].cpu(), 
                                     times=sample_flow_field_times[0].cpu(), 
                                     positions=sample_flow_field_positions[0].cpu(), 
                                     velocity=sample_flow_field_velocities[0].cpu(), 
                                     save_path=f'{root}/ground_truth_occupancy_and_flow.gif')
            
            render_flow_at_ground_truth_occupancy(model=model,
                                                  road_map=sample_road_map,
                                                  times=sample_flow_field_times,
                                                  positions=sample_flow_field_positions,
                                                  scene_context=scene_context,
                                                  save_path=f'{root}/estimated_flow_at_ground_truth_occupancy.gif')
            
            render_occupancy_and_flow_unoccluded(model=model, 
                                                 road_map=sample_road_map,
                                                 times=sample_flow_field_times, 
                                                 positions=sample_flow_field_positions, 
                                                 initial_time=11,
                                                 scene_context=scene_context,
                                                 save_path=f'{root}/estimated_occupancy_and_flow_unoccluded.gif')
        
            render_flow_field(model=model,
                              road_map=sample_road_map,
                              grid_size=sample_road_map[0].shape[0], 
                              stride=10, 
                              timesteps=91, 
                              freq=10,
                              scene_context=scene_context,
                              save_path=f'{root}/flow_field.gif')